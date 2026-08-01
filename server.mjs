// Minimal Square Connect screen.
// One page, one button: authorize the Square account, store the token locally,
// so the PyTorch sales model has something to pull orders from.
//
// Run: node server.mjs   ->   http://localhost:8080

import { createServer } from 'node:http';
import { randomBytes } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(fileURLToPath(import.meta.url));
const TOKEN_FILE = join(ROOT, '.square-tokens.json');

// --- env -------------------------------------------------------------------

function loadEnv() {
  const file = join(ROOT, '.env');
  if (!existsSync(file)) return;
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    const match = /^([A-Z0-9_]+)=(.*)$/.exec(line.trim());
    if (match && !process.env[match[1]]) process.env[match[1]] = match[2];
  }
}

loadEnv();

const APP_ID = process.env.SQUARE_APPLICATION_ID;
const APP_SECRET = process.env.SQUARE_APPLICATION_SECRET;
const ENVIRONMENT = process.env.SQUARE_ENVIRONMENT === 'production' ? 'production' : 'sandbox';
const REDIRECT_URL = process.env.SQUARE_REDIRECT_URL || 'http://localhost:8080/api/square/callback';
const PORT = Number(process.env.PORT || 8080);

const HOSTS =
  ENVIRONMENT === 'production'
    ? { authorize: 'https://connect.squareup.com', api: 'https://connect.squareup.com' }
    : { authorize: 'https://connect.squareupsandbox.com', api: 'https://connect.squareupsandbox.com' };

// Read-only scopes: everything collector/ pulls. Keep in sync with
// REQUIRED_SCOPES in collector/config.py — a missing scope here is a 403 there.
const SCOPES = [
  'MERCHANT_PROFILE_READ', // locations
  'ORDERS_READ',           // orders + line items
  'PAYMENTS_READ',         // payments + refunds
  'ITEMS_READ',            // catalog
  'CUSTOMERS_READ',        // customer directory
  'INVENTORY_READ',        // stock counts
  'EMPLOYEES_READ',        // team members
  'TIMECARDS_READ',        // labor shifts
];

// --- token storage ---------------------------------------------------------

async function readTokens() {
  try {
    return JSON.parse(await readFile(TOKEN_FILE, 'utf8'));
  } catch {
    return null;
  }
}

async function writeTokens(tokens) {
  await writeFile(TOKEN_FILE, JSON.stringify(tokens, null, 2), { mode: 0o600 });
}

// --- oauth -----------------------------------------------------------------

const pendingStates = new Set();

function authorizeUrl() {
  const state = randomBytes(24).toString('hex');
  pendingStates.add(state);
  const url = new URL('/oauth2/authorize', HOSTS.authorize);
  url.searchParams.set('client_id', APP_ID);
  url.searchParams.set('scope', SCOPES.join(' '));
  if (ENVIRONMENT === 'production') {
    url.searchParams.set('session', 'false');
  }
  url.searchParams.set('state', state);
  url.searchParams.set('redirect_uri', REDIRECT_URL);
  return url.toString();
}

async function exchangeCode(code) {
  const response = await fetch(`${HOSTS.api}/oauth2/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Square-Version': '2025-01-23' },
    body: JSON.stringify({
      client_id: APP_ID,
      client_secret: APP_SECRET,
      code,
      grant_type: 'authorization_code',
      redirect_uri: REDIRECT_URL,
    }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body?.errors?.[0]?.detail || 'Token exchange failed');
  return body;
}

async function fetchLocations(accessToken) {
  const response = await fetch(`${HOSTS.api}/v2/locations`, {
    headers: { Authorization: `Bearer ${accessToken}`, 'Square-Version': '2025-01-23' },
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body?.errors?.[0]?.detail || 'Could not list locations');
  return body.locations || [];
}

// --- responses -------------------------------------------------------------

function send(res, status, contentType, body) {
  res.writeHead(status, { 'Content-Type': contentType, 'Cache-Control': 'no-store' });
  res.end(body);
}

async function page(res, state) {
  const html = await readFile(join(ROOT, 'public', 'index.html'), 'utf8');
  send(res, 200, 'text/html; charset=utf-8', html.replace('__STATE__', JSON.stringify(state)));
}

// --- server ----------------------------------------------------------------

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  try {
    if (url.pathname === '/') {
      const tokens = await readTokens();
      return await page(res, {
        environment: ENVIRONMENT,
        applicationId: APP_ID,
        connected: Boolean(tokens?.access_token),
        merchantId: tokens?.merchant_id ?? null,
        locations: tokens?.locations ?? [],
        expiresAt: tokens?.expires_at ?? null,
      });
    }

    if (url.pathname === '/api/square/connect') {
      res.writeHead(302, { Location: authorizeUrl() });
      return res.end();
    }

    if (url.pathname === '/api/square/callback') {
      const error = url.searchParams.get('error');
      if (error) {
        return send(res, 400, 'text/html; charset=utf-8', errorPage(url.searchParams.get('error_description') || error));
      }
      const state = url.searchParams.get('state');
      const code = url.searchParams.get('code');
      if (!state || !pendingStates.delete(state)) {
        return send(res, 400, 'text/html; charset=utf-8', errorPage('State mismatch — start the connection again.'));
      }
      if (!code) return send(res, 400, 'text/html; charset=utf-8', errorPage('No authorization code returned.'));

      const tokens = await exchangeCode(code);
      const locations = await fetchLocations(tokens.access_token).catch(() => []);
      await writeTokens({
        ...tokens,
        environment: ENVIRONMENT,
        locations: locations.map((l) => ({ id: l.id, name: l.name, currency: l.currency })),
        connected_at: new Date().toISOString(),
      });
      res.writeHead(302, { Location: '/' });
      return res.end();
    }

    if (url.pathname === '/api/square/disconnect' && req.method === 'POST') {
      await writeFile(TOKEN_FILE, '{}', { mode: 0o600 });
      return send(res, 200, 'application/json', '{"ok":true}');
    }

    send(res, 404, 'text/plain', 'Not found');
  } catch (err) {
    send(res, 500, 'text/html; charset=utf-8', errorPage(err.message));
  }
});

function errorPage(message) {
  return `<!doctype html><meta charset="utf-8"><title>Square connection failed</title>
<body style="font:16px/1.5 -apple-system,system-ui,sans-serif;display:grid;place-items:center;height:100vh;margin:0;background:#0b0d10;color:#e7ecf2">
<div style="max-width:34rem;text-align:center">
<h1 style="font-size:1.25rem">Square connection failed</h1>
<p style="color:#9aa7b4">${message.replace(/[<&]/g, (c) => (c === '<' ? '&lt;' : '&amp;'))}</p>
<p><a href="/" style="color:#7cc2ff">Back</a></p>
</div></body>`;
}

if (!APP_ID || !APP_SECRET) {
  console.error('Missing SQUARE_APPLICATION_ID / SQUARE_APPLICATION_SECRET in .env');
  process.exit(1);
}

server.listen(PORT, () => {
  console.log(`Square connect screen: http://localhost:${PORT}  (${ENVIRONMENT})`);
  console.log(`Redirect URL must be registered in Square: ${REDIRECT_URL}`);
});
