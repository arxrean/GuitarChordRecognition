# GuitarChordRecognition
The goal of this project is to recognize the chord based on the input image.

### Dataset

Download the dataset from [here](https://drive.google.com/file/d/1D1ZNHSX_QuHGz_k0EEJ8sKJJBUy2eFR9/view?usp=sharing) and put it under the root project folder.

Create a `save` folder under the root project folder.

Below is the summary of our dataset.

|  Chord   | Total | Train | Validation | Test |
| :------: | :---: | :---: | :--------: | :--: |
|    C     |  60   |  40   |     10     |  10  |
|    Dm    |  60   |  40   |     10     |  10  |
|    Am    |  60   |  40   |     10     |  10  |
|    F     |  60   |  40   |     10     |  10  |
| Unseen-1 |  20   |   0   |     0      |  20  |
| Unseen-2 |  20   |   0   |     0      |  20  |

### Train

**Baseline**

A traditional classification model (resnet backbone + fc layer) to classify the images.

```python
python train.py --gpu --num_workers 0 --batch_size 32
```

**Joint Graph Model**

```python
python train_point.py --gpu --num_workers 0 --batch_size 3 --name point --dataset chord_point --middle point --decode point --batch_size 32
```

### Test

**Baseline**

```
python test.py --gpu --num_workers 0
```

### Result

**Seen chords**

|   Split    | Top-1 |
| :--------: | :---: |
| validation | 80.0% |
|    test    | 85.0% |

**Unseen chords**

TBD