def test_app_factory():
    from ai_tweets.serve import create_app
    app = create_app()
    assert app.title.startswith('LLM Text Classification')
