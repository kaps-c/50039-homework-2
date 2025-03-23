# 50039-homework-2

Celest Teng Roh Yee (1007036)

## Part 1

### Question 1
- Machine Learning Task (supervised classification task)
	- Supervised: CIFAR-10 and STL-10 are labelled datasets and since they contain 10 labelled classes it is a supervised learning problem.
	- Classification: "Image classification task" mentioned in the question, to assign each image input to one of the 10 categories. 
- Input features and Output classes
	- CIFAR-10: 3072 x 60000 = 184,320,000
		- No. of images: 60000
		- CIFAR-10 images (per image): 
			- width x height: 32 x 32
			- RGB channels: 3
			- Total input features: 32 x 32 x 3 = 3072
	- STL-10: 13000×27648 = 359424000
		- No. of images: 13000
		- STL-10 images (per image):
			- width x height: 96 x 96
			- RGB channels: 3
			- Total input features: 96 x 96 x 3 = 27684
	- Output classes: 10 classes
- What is the purpose of the `transforms.Resize((224, 224))` operation in the transform variable below? Why is it needed?
	- To make input compatible with pre-trained ResNet50 (which expects 224×224 images) - ResNet50 was trained on **ImageNet**, which uses **224×224×3** images. Without resizing, the model's convolutional and fully connected layers won’t match the input shape, which will result in errors.
### Question 2
How would you modify the architecture to work with the STL-10 Dataset with these two strategies? Write code to replace the missing part and show your code in your report.

```Python
model1 = resnet50(num_classes=10)
model1.load_state_dict(torch.load(pretrained_model_weight, map_location=device), strict=False)
model1.to(device)

# Freeze
for param in model1.parameters():
    param.requires_grad = False

model1.fc = torch.nn.Linear(2048, 10)

# train fc
for param in model1.fc.parameters():
    param.requires_grad = True

optimizer1 = optim.SGD(model1.fc.parameters(), lr=0.005, momentum=0.9)

# model2: fine-tune the entire model
model2 = resnet50(num_classes=10)
model2.load_state_dict(torch.load(pretrained_model_weight, map_location=device), strict=False)
model2.to(device)

optimizer2 = optim.SGD(None, lr=0.01, momentum=0.9)

for param in model2.parameters():
    param.requires_grad = True
```

### Question 3
Next, we will use two transfer learning strategies to retrain and test the models on the new dataset. Fill in the missing parts and show your code in your report.

```Python
num_epochs = 50
batch_size = 32

# Transforms (ResNet expects 224x224 images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Loss function
criterion = nn.CrossEntropyLoss()

# Logging
train_losses = {}
val_losses = {}
val_accuracies = {}

# Training + Evaluation Function
def train_and_evaluate(model, train_loader, test_loader, optimizer, num_epochs):
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(test_loader)
        accuracy = correct / total
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(accuracy)
        
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4%}"
        )
    return train_loss_list, val_loss_list, val_acc_list
```
### Question 4
Plot training/validation loss curves for both strategies.
- Compare their final test accuracies. Which strategy is more effective? Why or why not?
- Does unfreezing more layers always improve performance? Why or why not?
Strategy 1: 
Epoch (50/50), Train Loss: 1.4904, Val Loss: 1.4238, Val Acc: 50.2000%

Strategy 2
Epoch (50/50), Train Loss: 3.0714, Val Loss: 3.0708, Val Acc: 10.2000%

![[download.png]]
![[download (1).png]]

- Compare their final test accuracies. Which strategy is more effective? Why or why not?
	- Strategy 1 was more effective. Referring to the graph, strategy 1 is still training, as loss decreases over time and accuracy increases. However, for strategy 2, loss and accuracy stagnate (accuracy at 10% approximately, which is no better than just random guessing).
	- Strategy 1 performed better because it kept useful features from the pretrained Resnet model. 
	- Strategy 2 did not work out because the dataset was too small (which messed with the pre-trained Resnet weights) + the learning rate was too high (0.01) and can be reduced. 
- Does unfreezing more layers always improve performance? Why or why not?
	- Not always, since unfreezing more layers means allowing more parameters to be changed during training, which means that:
		- The model may take longer to train (more parameters)
		- If the model has too little data, will lead to overfitting on the dataset :(
	- Also, the supervised classification task solved by our models was sufficiently similar to the problem solved by Resnet, so there was no need to unfreeze that many layers in the first place.
### Question 5

```Python
"""Data Augmentation and Preprocessing"""

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),   # Flip horizontally with probability 0.5
    transforms.RandomRotation(degrees=15),    # Rotate randomly within ±15 degrees
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

### Question 6
Explain why data augmentation is critical for small datasets. How might these transforms improve model generalization? Are the transformations we suggested in this notebook appropriate? Would you suggest using different ones?

- Explain why data augmentation is critical for small datasets. 
	- Small datasets may not capture full range of real world variations, e.g. different orientations, flipped, etc.
	- Data augmentation artificially increases size of dataset by applying transformations on the data, helping to reduce overfitting (due to small dataset), leading to a more robust model. 
- How might these transforms improve model generalization?
	- `RandomHorizontalFlip(p=0.5)`: Makes the model more robust and able to classify correctly when there is change in left/right orientation. Useful for symmetric objects like animals, vehicles, etc.
	- `RandomRotation(degrees=15)`: Allows model to handle tilted objects.
	- Exposes model to different versions of same image, so that the model does not learn patterns by relying on specific pixel arrangements or specific patterns in the order of the dataset, but rather recognizes general patterns so that it can generalize better to unseen data. 
- Are the transformations we suggested in this notebook appropriate? Would you suggest using different ones?
	- Yes, they are appropriate since:
		- The class should remain the same even with these transformations (on everyday objects like cars in CIFAR-10).
		- The model should be therefore able to recognize these objects even with these transformations (especially important for generalizing since the images in CIFAR-10 are of everyday objects that are frequently photographed in different orientations)

### Question 7
Can you suggest another way to perform transfer learning on these models that might be more effective than the current implementation? What would you do differently? Extra points will be given based on effort and performance of the proposed approach. You might want to consider dropping the `transforms.Resize((224, 224))` operation from earlier and adjust/retrain some layers from the ResNet architecture instead as shown in class on W4S3.

- Can you suggest another way to perform transfer learning on these models that might be more effective than the current implementation?
	- Progressive unfreezing to progressively retrain layers
- What would you do differently? 
	- Modify first convolution layer of Resnet since Resnet was used on ImageNet which has images that are 224x224, but STL-10 images are 96x96. Hence, drop the `transforms.Resize((224, 224))`.
 
 ```Python
model = resnet50(num_classes=10)

# edit first conv layer to work better with 96x96 images
model.conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

model.to(device)
```
	  
	- Gradually unfreeze instead: Train only fc first, then gradually unfreeze more, so that model can progressively adapt to STL-10. 

## Part 2
### Question 1 

(1) Fractions have been simplified.

| 7/9 | 5/9 | -5/9 |
|-----|-----|------|
| 2/3 | 2/3 | -1/9 |
| 0   | 1/3 | 2/3  |

(2) Using zero padding as specified in the notes attached.

| 5/9 | 0   | 2/9 |
|-----|-----|-----|
| 0   | 2/3 | 1/9 |
| 0   | 4/9 | 1/3 |

### Question 2

`Softmax` function:

$$
\text{softmax}(h_i) = \frac{e^{h_i}}{\sum_{j=1}^{3} e^{h_j}}
$$

`Softmax` probabilities:

$$
p_1 = \frac{8.17}{10.4555} \approx 0.7817 \\
p_2 = \frac{0.2725}{10.4555} \approx 0.0261 \\
p_3 = \frac{2.013}{10.4555} \approx 0.1923
$$


CE loss:

$$
L = -\log(p_{\text{true class}})
$$
$$
L = -\ln(0.7817) \approx 0.246
$$
$$
\boxed{L \approx 0.246}
$$

### Question 3
Formula: 
$$
\text{Parameters} = (K_w \times K_h \times C_{\text{in}} \times C_{\text{out}}) + C_{\text{out}}
$$

For `AlexNet`, which is the diagram shown: 

**Conv1:**

$$
(11 \times 11) \times 3 \times 96 + 96 = 34,944
$$

**Conv2:**

$$
(5 \times 5) \times 96 \times 256 + 256 = 614,656
$$

**Conv3:**

$$
(3 \times 3) \times 256 \times 384 + 384 = 885,120
$$

**Conv4:**

$$
(3 \times 3) \times 384 \times 384 + 384 = 1,327,488
$$

**Conv5:**

$$
(3 \times 3) \times 384 \times 256 + 256 = 884,992
$$

**FC1:**

$$
(6 \times 6) \times 256 \times 4096 + 4096 = 37,752,832
$$

**FC2:**

$$
4096 \times 4096 + 4096 = 16,781,312
$$

**FC3 (`softmax` output):**

$$
4096 \times 1000 + 1000 = 4,097,000
$$

**Total:**

$$
34,944 + 614,656 + 885,120 + 1,327,488 + 884,992 + 37,752,832 + 16,781,312 + 4,097,000 = \boxed{62,378,344}
$$
