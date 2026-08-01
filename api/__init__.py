"""Phase 6: serve the model over HTTP.

    collector -> database -> pipelines -> training -> models -> api -> dashboard

The API owns no machine learning. It loads what training/ saved and answers
questions about it.

    uvicorn api.main:app --reload
"""
