def test_train_importable():
    from ai_tweets.train import train
    assert callable(train)
