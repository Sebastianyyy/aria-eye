# aria-eye


## DATA

Downloading dataset

1) cd ./data

2) ./download_data_train.sh

3) ./download_data_test.sh

4) Basic usage

```
    aria = AriaDataset(f"./data/downloads_test/",sample=10,frame_grabber=2)

    test_dataloader = DataLoader(aria, batch_size=2, shuffle=False,drop_last=True) 
```

