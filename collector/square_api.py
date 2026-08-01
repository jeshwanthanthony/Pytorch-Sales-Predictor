"""Pull every entity Square will give us, flattened into training-ready rows.

Each `fetch_*` returns an iterator of plain dicts whose keys are the columns the
database will get. Money stays in Square's smallest denomination (cents, int) —
converting to dollars is a feature-engineering decision, not a storage one.

Timestamps stay as Square's RFC 3339 UTC strings. Local calendar dates are
derived once, in features, using the location's timezone.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Iterator

from .config import SQUARE_VERSION, SquareAuth
from .http import ApiError, HttpClient, paginate

log = logging.getLogger(__name__)

# square allows 1000 orders per page, the other endpoints only 100-200
ORDERS_PAGE = 500
DEFAULT_PAGE = 100


def _money(value: dict[str, Any] | None) -> int:
    """Square Money -> integer cents. Absent money means zero, not null."""
    if not value:
        return 0
    return int(value.get("amount") or 0)


def _currency(value: dict[str, Any] | None) -> str | None:
    return (value or {}).get("currency")


def _rfc3339(value: date | datetime) -> str:
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    else:
        moment = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SquareCollector:
    def __init__(self, auth: SquareAuth, client: HttpClient | None = None):
        self.auth = auth
        self._client = client or HttpClient(
            base_url=auth.api_host,
            headers={
                "Authorization": f"Bearer {auth.access_token}",
                "Content-Type": "application/json",
                "Square-Version": SQUARE_VERSION,
            },
        )

    def __enter__(self) -> SquareCollector:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.__exit__(*exc)

    # -- locations ----------------------------------------------------------

    def location_ids(self) -> list[str]:
        """Locations from the token file, or a live lookup if it had none."""
        if self.auth.location_ids:
            return self.auth.location_ids
        # fetch_locations returns normalized rows, so the key is location_id not id
        return [loc["location_id"] for loc in self.fetch_locations() if loc.get("location_id")]

    def fetch_locations(self) -> list[dict[str, Any]]:
        body = self._client.get("/v2/locations")
        rows = []
        for loc in body.get("locations") or []:
            address = loc.get("address") or {}
            coords = loc.get("coordinates") or {}
            rows.append(
                {
                    "location_id": loc.get("id"),
                    "name": loc.get("name"),
                    "status": loc.get("status"),
                    "currency": loc.get("currency"),
                    "country": address.get("country"),
                    "state": address.get("administrative_district_level_1"),
                    "city": address.get("locality"),
                    "postal_code": address.get("postal_code"),
                    "address_line_1": address.get("address_line_1"),
                    "latitude": coords.get("latitude"),
                    "longitude": coords.get("longitude"),
                    "timezone": loc.get("timezone"),
                    "business_name": loc.get("business_name"),
                    "type": loc.get("type"),
                    "created_at": loc.get("created_at"),
                }
            )
        return rows

    # -- orders + line items ------------------------------------------------

    def search_orders(self, start: date | datetime, end: date | datetime) -> Iterator[dict[str, Any]]:
        """Raw order payloads updated in the window.

        Filters on updated_at rather than created_at so an order modified after
        the fact (comped, refunded, reopened) is re-collected.
        """
        location_ids = self.location_ids()
        if not location_ids:
            log.warning("no locations available; skipping orders")
            return

        # square only takes 10 location ids at a time
        for chunk_start in range(0, len(location_ids), 10):
            chunk = location_ids[chunk_start : chunk_start + 10]

            def fetch(cursor: str | None, chunk: list[str] = chunk) -> dict[str, Any]:
                payload: dict[str, Any] = {
                    "location_ids": chunk,
                    "limit": ORDERS_PAGE,
                    "query": {
                        "filter": {
                            "date_time_filter": {
                                "updated_at": {"start_at": _rfc3339(start), "end_at": _rfc3339(end)}
                            }
                        },
                        "sort": {"sort_field": "UPDATED_AT", "sort_order": "ASC"},
                    },
                }
                if cursor:
                    payload["cursor"] = cursor
                return self._client.post("/v2/orders/search", json=payload)

            yield from paginate(fetch, "orders")

    def fetch_orders(self, start: date | datetime, end: date | datetime) -> Iterator[dict[str, Any]]:
        """One normalized row per order."""
        for order in self.search_orders(start, end):
            yield self.normalize_order(order)

    @staticmethod
    def normalize_order(order: dict[str, Any]) -> dict[str, Any]:
        totals = {
            "revenue_cents": _money(order.get("total_money")),
            "discount_cents": _money(order.get("total_discount_money")),
            "tax_cents": _money(order.get("total_tax_money")),
            "tip_cents": _money(order.get("total_tip_money")),
            "service_charge_cents": _money(order.get("total_service_charge_money")),
        }
        # net sales is what the kitchen actually sold, before tax and tip
        totals["net_sales_cents"] = (
            totals["revenue_cents"] - totals["tax_cents"] - totals["tip_cents"]
        )

        fulfillments = order.get("fulfillments") or []
        tenders = order.get("tenders") or []
        returns = order.get("returns") or []

        customer_id = order.get("customer_id")
        if not customer_id:
            for fulfillment in fulfillments:
                recipient = (
                    (fulfillment.get("pickup_details") or {}).get("recipient")
                    or (fulfillment.get("delivery_details") or {}).get("recipient")
                    or {}
                )
                if recipient.get("customer_id"):
                    customer_id = recipient["customer_id"]
                    break

        return {
            "order_id": order.get("id"),
            "location_id": order.get("location_id"),
            "created_at": order.get("created_at"),
            "updated_at": order.get("updated_at"),
            "closed_at": order.get("closed_at"),
            "state": order.get("state"),
            "currency": _currency(order.get("total_money")),
            **totals,
            "source": (order.get("source") or {}).get("name"),
            "fulfillment_type": fulfillments[0].get("type") if fulfillments else None,
            "fulfillment_state": fulfillments[0].get("state") if fulfillments else None,
            "customer_id": customer_id,
            "payment_types": sorted({t.get("type") for t in tenders if t.get("type")}),
            "tender_count": len(tenders),
            "line_item_count": len(order.get("line_items") or []),
            "item_quantity": sum(
                float(item.get("quantity") or 0) for item in order.get("line_items") or []
            ),
            "has_returns": bool(returns),
            "ticket_name": order.get("ticket_name"),
            "version": order.get("version"),
        }

    @staticmethod
    def fetch_order_items(orders: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Explode raw order payloads into one row per line item.

        Takes the *raw* payloads from search_orders, not the normalized rows —
        line items only exist on the original response.
        """
        for order in orders:
            for index, item in enumerate(order.get("line_items") or []):
                modifiers = item.get("modifiers") or []
                yield {
                    "order_id": order.get("id"),
                    "line_item_uid": item.get("uid"),
                    "line_number": index,
                    "location_id": order.get("location_id"),
                    "created_at": order.get("created_at"),
                    "catalog_object_id": item.get("catalog_object_id"),
                    "item_name": item.get("name"),
                    "variation_name": item.get("variation_name"),
                    # square does not put the category on a line item, we join it later
                    "quantity": float(item.get("quantity") or 0),
                    "unit": ((item.get("quantity_unit") or {}).get("measurement_unit") or {}).get(
                        "type"
                    ),
                    "base_price_cents": _money(item.get("base_price_money")),
                    "gross_sales_cents": _money(item.get("gross_sales_money")),
                    "discount_cents": _money(item.get("total_discount_money")),
                    "tax_cents": _money(item.get("total_tax_money")),
                    "total_cents": _money(item.get("total_money")),
                    "modifiers": [
                        {
                            "uid": mod.get("uid"),
                            "catalog_object_id": mod.get("catalog_object_id"),
                            "name": mod.get("name"),
                            "price_cents": _money(mod.get("base_price_money")),
                        }
                        for mod in modifiers
                    ],
                    "modifier_names": [m.get("name") for m in modifiers if m.get("name")],
                    "item_type": item.get("item_type"),
                    "note": item.get("note"),
                }

    # -- payments -----------------------------------------------------------

    def fetch_payments(self, start: date | datetime, end: date | datetime) -> Iterator[dict[str, Any]]:
        def fetch(cursor: str | None) -> dict[str, Any]:
            params: dict[str, Any] = {
                "begin_time": _rfc3339(start),
                "end_time": _rfc3339(end),
                "sort_order": "ASC",
                "limit": DEFAULT_PAGE,
            }
            if cursor:
                params["cursor"] = cursor
            return self._client.get("/v2/payments", params=params)

        for payment in paginate(fetch, "payments"):
            card = payment.get("card_details") or {}
            card_info = card.get("card") or {}
            yield {
                "payment_id": payment.get("id"),
                "order_id": payment.get("order_id"),
                "location_id": payment.get("location_id"),
                "customer_id": payment.get("customer_id"),
                "created_at": payment.get("created_at"),
                "updated_at": payment.get("updated_at"),
                "status": payment.get("status"),
                "amount_cents": _money(payment.get("amount_money")),
                "tip_cents": _money(payment.get("tip_money")),
                "app_fee_cents": _money(payment.get("app_fee_money")),
                "refunded_cents": _money(payment.get("refunded_money")),
                "approved_cents": _money(payment.get("approved_money")),
                "currency": _currency(payment.get("amount_money")),
                "processing_fee_cents": sum(
                    _money(fee.get("amount_money")) for fee in payment.get("processing_fee") or []
                ),
                # CARD / CASH / GIFT_CARD / EXTERNAL ...
                "source_type": payment.get("source_type"),
                "card_brand": card_info.get("card_brand"),
                "card_type": card_info.get("card_type"),
                "last_4": card_info.get("last_4"),
                "entry_method": card.get("entry_method"),
                "receipt_number": payment.get("receipt_number"),
                "team_member_id": payment.get("team_member_id"),
            }

    # -- refunds ------------------------------------------------------------

    def fetch_refunds(self, start: date | datetime, end: date | datetime) -> Iterator[dict[str, Any]]:
        def fetch(cursor: str | None) -> dict[str, Any]:
            params: dict[str, Any] = {
                "begin_time": _rfc3339(start),
                "end_time": _rfc3339(end),
                "sort_order": "ASC",
                "limit": DEFAULT_PAGE,
            }
            if cursor:
                params["cursor"] = cursor
            return self._client.get("/v2/refunds", params=params)

        for refund in paginate(fetch, "refunds"):
            yield {
                "refund_id": refund.get("id"),
                "payment_id": refund.get("payment_id"),
                "order_id": refund.get("order_id"),
                "location_id": refund.get("location_id"),
                "created_at": refund.get("created_at"),
                "updated_at": refund.get("updated_at"),
                "status": refund.get("status"),
                "amount_cents": _money(refund.get("amount_money")),
                "currency": _currency(refund.get("amount_money")),
                "processing_fee_cents": sum(
                    _money(fee.get("amount_money")) for fee in refund.get("processing_fee") or []
                ),
                "reason": refund.get("reason"),
                "destination_type": refund.get("destination_type"),
                "team_member_id": refund.get("team_member_id"),
            }

    # -- customers ----------------------------------------------------------

    def fetch_customers(self) -> Iterator[dict[str, Any]]:
        """The customer directory.

        Visit count and lifetime spend are deliberately absent: Square doesn't
        expose them, and deriving them from our own order history is both more
        accurate and free.
        """

        def fetch(cursor: str | None) -> dict[str, Any]:
            params: dict[str, Any] = {"limit": DEFAULT_PAGE, "sort_field": "CREATED_AT"}
            if cursor:
                params["cursor"] = cursor
            return self._client.get("/v2/customers", params=params)

        for customer in paginate(fetch, "customers"):
            preferences = customer.get("preferences") or {}
            yield {
                "customer_id": customer.get("id"),
                "created_at": customer.get("created_at"),
                "updated_at": customer.get("updated_at"),
                "given_name": customer.get("given_name"),
                "family_name": customer.get("family_name"),
                "email": customer.get("email_address"),
                "phone": customer.get("phone_number"),
                "birthday": customer.get("birthday"),
                "reference_id": customer.get("reference_id"),
                "company_name": customer.get("company_name"),
                "creation_source": customer.get("creation_source"),
                "group_ids": customer.get("group_ids") or [],
                "segment_ids": customer.get("segment_ids") or [],
                "email_unsubscribed": preferences.get("email_unsubscribed"),
                "postal_code": (customer.get("address") or {}).get("postal_code"),
                "note": customer.get("note"),
            }

    # -- catalog ------------------------------------------------------------

    def fetch_catalog(self) -> Iterator[dict[str, Any]]:
        """Items, their variations, and the category each belongs to.

        Categories are pulled in the same request so item rows carry a readable
        category name instead of an opaque id.
        """
        objects: list[dict[str, Any]] = []

        def fetch(cursor: str | None) -> dict[str, Any]:
            params: dict[str, Any] = {"types": "ITEM,CATEGORY,MODIFIER_LIST"}
            if cursor:
                params["cursor"] = cursor
            return self._client.get("/v2/catalog/list", params=params)

        objects.extend(paginate(fetch, "objects"))

        categories = {
            obj["id"]: (obj.get("category_data") or {}).get("name")
            for obj in objects
            if obj.get("type") == "CATEGORY"
        }

        for obj in objects:
            if obj.get("type") != "ITEM":
                continue
            data = obj.get("item_data") or {}
            category_id = data.get("category_id") or (data.get("categories") or [{}])[0].get("id")
            variations = data.get("variations") or []
            yield {
                "catalog_object_id": obj.get("id"),
                "item_name": data.get("name"),
                "description": data.get("description_plaintext") or data.get("description"),
                "category_id": category_id,
                "category_name": categories.get(category_id),
                "product_type": data.get("product_type"),
                "is_archived": data.get("is_archived", False),
                "is_deleted": obj.get("is_deleted", False),
                "updated_at": obj.get("updated_at"),
                "version": obj.get("version"),
                "modifier_list_ids": [
                    info.get("modifier_list_id")
                    for info in data.get("modifier_list_info") or []
                    if info.get("modifier_list_id")
                ],
                "variations": [
                    {
                        "id": variation.get("id"),
                        "name": (variation.get("item_variation_data") or {}).get("name"),
                        "sku": (variation.get("item_variation_data") or {}).get("sku"),
                        "price_cents": _money(
                            (variation.get("item_variation_data") or {}).get("price_money")
                        ),
                    }
                    for variation in variations
                ],
            }

    # -- inventory ----------------------------------------------------------

    def fetch_inventory(self, catalog_object_ids: list[str]) -> Iterator[dict[str, Any]]:
        """Current stock counts for the given catalog objects (usually variations)."""
        if not catalog_object_ids:
            return

        location_ids = self.location_ids()
        # the batch endpoint takes up to 1000 ids at once
        for start in range(0, len(catalog_object_ids), 500):
            batch = catalog_object_ids[start : start + 500]

            def fetch(cursor: str | None, batch: list[str] = batch) -> dict[str, Any]:
                payload: dict[str, Any] = {"catalog_object_ids": batch, "location_ids": location_ids}
                if cursor:
                    payload["cursor"] = cursor
                return self._client.post("/v2/inventory/counts/batch-retrieve", json=payload)

            for count in paginate(fetch, "counts"):
                yield {
                    "catalog_object_id": count.get("catalog_object_id"),
                    "catalog_object_type": count.get("catalog_object_type"),
                    "location_id": count.get("location_id"),
                    "state": count.get("state"),
                    "quantity": float(count.get("quantity") or 0),
                    "calculated_at": count.get("calculated_at"),
                }

    # -- team + labor -------------------------------------------------------

    def fetch_team_members(self) -> Iterator[dict[str, Any]]:
        def fetch(cursor: str | None) -> dict[str, Any]:
            payload: dict[str, Any] = {"limit": 200}
            if cursor:
                payload["cursor"] = cursor
            return self._client.post("/v2/team-members/search", json=payload)

        for member in paginate(fetch, "team_members"):
            assignment = member.get("assigned_locations") or {}
            yield {
                "team_member_id": member.get("id"),
                "reference_id": member.get("reference_id"),
                "status": member.get("status"),
                "given_name": member.get("given_name"),
                "family_name": member.get("family_name"),
                "is_owner": member.get("is_owner", False),
                "created_at": member.get("created_at"),
                "updated_at": member.get("updated_at"),
                "assignment_type": assignment.get("assignment_type"),
                "location_ids": assignment.get("location_ids") or [],
            }

    def fetch_shifts(self, start: date | datetime, end: date | datetime) -> Iterator[dict[str, Any]]:
        """Worked shifts — the ground truth for a future labor model."""

        def fetch(cursor: str | None) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "limit": 200,
                "query": {
                    "filter": {
                        "start": {"start_at": _rfc3339(start), "end_at": _rfc3339(end)},
                    },
                    "sort": {"field": "START_AT", "order": "ASC"},
                },
            }
            if cursor:
                payload["cursor"] = cursor
            return self._client.post("/v2/labor/shifts/search", json=payload)

        for shift in paginate(fetch, "shifts"):
            wage = shift.get("wage") or {}
            hourly = _money(wage.get("hourly_rate"))
            start_at, end_at = shift.get("start_at"), shift.get("end_at")
            hours = _hours_between(start_at, end_at)
            yield {
                "shift_id": shift.get("id"),
                "team_member_id": shift.get("team_member_id"),
                "location_id": shift.get("location_id"),
                "start_at": start_at,
                "end_at": end_at,
                "hours": hours,
                "status": shift.get("status"),
                "job_id": wage.get("job_id"),
                "job_title": wage.get("title"),
                "hourly_rate_cents": hourly,
                "labor_cost_cents": round(hourly * hours) if hours else 0,
                "declared_tips_cents": _money(shift.get("declared_cash_tip_money")),
                "timezone": shift.get("timezone"),
            }


def _hours_between(start_at: str | None, end_at: str | None) -> float:
    """Shift length in hours; 0.0 for an open shift that has no end yet."""
    if not start_at or not end_at:
        return 0.0
    try:
        start = datetime.fromisoformat(start_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max((end - start).total_seconds() / 3600, 0.0)


def probe_scopes(collector: SquareCollector) -> dict[str, bool]:
    """Which pulls this token is actually allowed to make.

    Cheap one-row calls per endpoint, so a missing scope surfaces as a clear
    report at the top of a run instead of a 403 twenty minutes in.
    """
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    checks = {
        "locations": lambda: collector.fetch_locations(),
        "orders": lambda: next(collector.fetch_orders(yesterday, now), None),
        "payments": lambda: next(collector.fetch_payments(yesterday, now), None),
        "refunds": lambda: next(collector.fetch_refunds(yesterday, now), None),
        "customers": lambda: next(collector.fetch_customers(), None),
        "catalog": lambda: next(collector.fetch_catalog(), None),
        "team_members": lambda: next(collector.fetch_team_members(), None),
        "shifts": lambda: next(collector.fetch_shifts(yesterday, now), None),
    }

    results: dict[str, bool] = {}
    for name, check in checks.items():
        try:
            check()
            results[name] = True
        except ApiError as exc:
            if exc.status in (401, 403):
                results[name] = False
            else:
                log.warning("probe %s: %s", name, exc)
                results[name] = False
    return results
