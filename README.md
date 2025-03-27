# 🔤 Sign Language Classification with PyTorch Lightning

This project implements classification of American Sign Language (ASL) letters using the Sign Language MNIST dataset and `PyTorch Lightning`.

## 📁 Contents

- ✅ Dataset download and extraction
- ✅ Custom `torch.utils.data.Dataset`
- ✅ `LightningDataModule` implementation
- ✅ `LightningModule` model definition
- ✅ Model training with checkpoint saving
- ✅ Inference on a single image
- ✅ `--fast_dev_run` argument for quick test run

## 🚀 How to Run

Install dependencies:

```bash
pip install pytorch-lightning pandas numpy torch torchvision
```

Run the main script:

```bash
python M1_PyTorchLightining_practice.py
```

Test run mode:

```bash
python M1_PyTorchLightining_practice.py --fast_dev_run True
```

## 🧠 Dataset

We use the [Sign Language MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist), which is automatically downloaded and extracted when the script runs.

## 🐳 Docker (Optional)

If you prefer not to install dependencies manually, you can use Docker:

```bash
docker build -t lightning-mnist .
docker run --rm -it lightning-mnist
```

Run with `--fast_dev_run` flag:

```bash
docker run --rm -it lightning-mnist python M1_PyTorchLightining_practice.py --fast_dev_run True
```

---

## 🧾 License

MIT License
