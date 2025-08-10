# Algorithms Developed for Stroke Lesion Detection

The heterogeneous morphology of stroke lesions and the low tissue contrast pose significant challenges for traditional image processing methods. This study aims to overcome these challenges using modern statistical approaches, topological data analysis, and AI-supported models.

<img width="367" height="157" alt="resim_2025-08-10_130229340" src="https://github.com/user-attachments/assets/bfa679d9-326f-41e3-8f93-90626af4d451" />
<img width="204" height="196" alt="resim_2025-08-10_130323448" src="https://github.com/user-attachments/assets/fd6fe19c-2102-4566-bd15-a23661014f3d" />



Four different algorithms were developed based on the state-of-the-art approaches discussed previously:

- **First algorithm:** Uses topological analysis to measure the symmetric difference between the left and right brain hemispheres, producing a meaningful numerical feature.
- **Second algorithm:** Employs Modified Gram-Schmidt (MGS) procedures.
- **Third algorithm:** Utilizes Bayesian statistical models.
- **Fourth algorithm:** Incorporates a data augmentation process applying various transformations to images to enhance model generalization.

---

## Dataset Preparation

Our project used two different datasets containing approximately **900,000** CT images in total:

1. **Ministry of Health of Turkey Open Data Portal**  
   Stroke dataset from the 2021 AI in Health competition, consisting of 6,651 training and 200 test images.

2. **Radiological Society of North America (RSNA) Brain Hemorrhage CT Dataset**  
   The world’s largest publicly available brain hemorrhage CT dataset with 874,035 images, collected from multiple institutions and countries [8].

For the comparison phase, all algorithms were trained on the Ministry of Health dataset and their outputs analyzed comparatively. The algorithm achieving the highest F1 score was retrained including the RSNA dataset.

A custom `CTDataset` class was developed to obtain consistent and processable data formats during training and testing. This class systematically processes images into proper tensor formats:

- Images are loaded using the PIL library, converted to grayscale (L mode), and resized to 512×512 pixels using bilinear interpolation.
- Invalid or corrupted files are logged and excluded.
- Suitable augmentation pipelines are applied, and finally, images are converted to float tensors while labels are converted to long tensors.

---

## Algorithms Used

### Topological Elliptical Symmetry Analysis for Asymmetry Feature Extraction

An elliptical symmetry transformation is applied to extract structural asymmetry features from medical images. The process includes:

1. **Gaussian Smoothing and Thresholding:**  
   A 2D Gaussian convolution filter with standard deviation \( \sigma = 5 \) is applied to the input image, followed by segmentation at different grayscale thresholds \( T \):

   \[
   S(x,y) = \begin{cases}
   1, & \text{if } T_{\text{low}} \leq \text{grayscale}(x,y) \leq T_{\text{high}} \\
   0, & \text{otherwise}
   \end{cases}
   \]

2. **Morphological Operations:**  
   Morphological closing is applied using a 5×5 structuring element to merge regions.

3. **Ellipse Fitting:**  
   The largest connected component is identified and a minimum bounding ellipse is estimated via least squares fitting.

4. **Hemisphere Segmentation:**  
   The symmetry axis is defined by the normal vector perpendicular to the main axis:

   \[
   \mathbf{n} = \big(\cos(\theta + \pi/2), \sin(\theta + \pi/2)\big)
   \]

5. **HU-Based Asymmetry Features:**  
   After thresholding Hounsfield Units between 50-100 HU, pixel-wise differences create an absolute asymmetry map:

   \[
   D(x,y) = | HL(x,y) - HR(x,y) |
   \]

6. **Regional Thresholding and Difference Calculation:**  
   Differences between the left hemisphere and the mirrored right hemisphere are computed using Mean Squared Difference (MSD) metrics. Optimal threshold pairs are chosen based on the lowest p-values.

---

### Modified Gram-Schmidt (MGS) Procedure

The first step in model training is applying the modified Gram-Schmidt algorithm to orthogonalize input weights:

- Given weight matrix \( W \in \mathbb{R}^{m \times n} \) where \( m \) and \( n \) are the number of output and input features respectively.
- MGS performs QR decomposition:

\[
W = QR
\]

where \( Q \) is orthogonal (\( Q^T Q = I \)) and \( R \) is upper triangular.

- Network parameters are initialized as follows:
  1. Weight matrices \( W_1 \in \mathbb{R}^{h \times d} \) and \( W_2 \in \mathbb{R}^{c \times h} \) are sampled from a Gaussian distribution:

  \[
  W_i \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 0.01
  \]

  2. Bias vectors \( b_1 \in \mathbb{R}^{h \times 1} \) and \( b_2 \in \mathbb{R}^{c \times 1} \) are initialized as zeros.

- Forward propagation:

\[
Z_1 = W_1 X + b_1, \quad A_1 = \max(0, Z_1)
\]

Here, \( A_1 \) is the output after applying ReLU activation.

\[
Z_2 = W_2 A_1 + b_2, \quad A_2 = \text{softmax}(Z_2)
\]

Softmax function per component:

\[
A_{2,i} = \frac{e^{Z_{2,i}}}{\sum_j e^{Z_{2,j}}}
\]

- The network is optimized using cross-entropy loss, with gradients computed via backpropagation.

---

### Bayesian Statistical Approach

- The model is based on **ResNet-50** architecture, where the first and last fully-connected layers are replaced with Bayesian layers.
- **Bayesian Linear Layers** use prior distributions for weights, enabling learning under uncertainty.
- Kullback-Leibler (KL) divergence between prior and posterior distributions regularizes the training:

\[
D_{KL}(q \parallel p) = \sum_i \log \frac{\sigma_{\text{prior}}}{\sigma_i} + \frac{\sigma_i^2 + \mu_i^2}{2 \sigma_{\text{prior}}^2} - \frac{1}{2}
\]

Here, \( \mu_i \) and \( \sigma_i \) denote the mean and standard deviation of weights; \( \sigma_{\text{prior}} \) is the prior standard deviation.

- **Bayesian Convolutional Layers** replace traditional convolutional layers; weights and biases’ uncertainties are modeled via reparameterization.
- Training uses stochastic gradient descent-based SGLD optimizer.
- Total loss function:

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{nll}} + \lambda \cdot D_{KL}
\]

Where \( \mathcal{L}_{\text{nll}} \) is negative log-likelihood, and \( \lambda \) controls KL term weighting.
- Model accuracy is calculated over multiple forward passes to estimate uncertainty.

---

### Model Architecture and Optimization Methods

- Various deep learning architectures evaluated: ResNet, DenseNet, EfficientNet, Transformer-based models.
- Data imbalance in CT data addressed with AdamW optimizer and preprocessing like Gaussian blur.
- Original RGB models modified for single-channel CT input by changing first convolution layer to accept 1 channel.
- Final fully connected layer replaced with a 2-class output layer (normal vs. stroke).
- Data augmentation pipeline designed to increase variation and prevent overfitting using **Albumentations** library:

  1. **GaussianBlur (p=0.3):** 30% chance of applying blur filter:

  \[
  G(x,y) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
  \]

  2. **RandomRotate90 (p=0.5):** 50% chance to rotate images in 90° increments.
  3. **HorizontalFlip & VerticalFlip (p=0.5):** Random flips to add symmetric variations.
  4. **Normalize & ToTensorV2:** Normalize images (mean=0.485, std=0.229) and convert to tensor.

- Training monitored in real-time with TQDM; Python logging module records errors and performance.
- Mixed precision training via `torch.cuda.amp.autocast()` and gradient scaling (`GradScaler`) improve speed and stability.
- Optimizer: **AdamW** with learning rate 1e-4.
- Loss: **CrossEntropyLoss**.
- Best model per epoch saved as checkpoint.
- Validation uses only normalization and tensor conversion for consistent performance.

---

## Thinking Like a Doctor

Doctors often use the brain's symmetrical anatomy to detect pathology by comparing with healthy sides. To emulate this, we developed **topological elliptical symmetry analysis**:

- An ellipse enclosing the skull’s outer contour is drawn.
- A symmetry curve along the ellipse's major axis is drawn.
- The right hemisphere is mirrored and overlaid onto the left hemisphere.
- The squared difference of grayscale values between hemispheres is computed.
- To reduce noise, we researched optimal lower and upper grayscale bounds emphasizing hyperdense hemorrhagic regions.
- All other grayscale values are set to zero.

![Topological Elliptical Symmetry Analysis for Asymmetry Feature Extraction](link_to_figure_2)

---

## Data Augmentation and Optimization

- Multiple algorithms were trained on the dataset using high-performance computing (HPC) resources.
- Models’ accuracy and training times were compared in tabular form.
- The strongest models were re-evaluated with larger datasets to find the most suitable one.
- To prevent overfitting and increase data variability, CT slices underwent Gaussian blur, random rotations, horizontal and vertical flips.
- AdamW optimizer was used to improve weight decay and regularization effectiveness.

---

# References

[1] Suzuki, K. (2017). Overview of deep learning in medical imaging. *Radiological Physics and Technology*, 10(3), 257-273.

[2] Ozaltin, O., Coskun, O., Yeniay, O., & Subasi, A. (2022). A deep learning approach for detecting stroke from brain CT images using OzNet. *Bioengineering*, 9(12), 783.

[3] Krishnan, R., Subedar, M., & Tickoo, O. (2020, April). Specifying weight priors in bayesian deep neural networks with empirical bayes. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 34, No. 04, pp. 4477-4484).

[4] Webber, G., & Reader, A. J. (2024). Diffusion models for medical image reconstruction. *BJR| Artificial Intelligence*, 1(1), ubae013.

[5] Faye, E. C., Fall, M. D., & Dobigeon, N. (2024). Regularization by denoising: Bayesian model and Langevin-within-split. *IEEE Transactions on Image Processing*.

[6] Singh, Y., Farrelly, C. M., Hathaway, Q. A., Leiner, T., Jagtap, J., Carlsson, G. E., & Erickson, B. J. (2023). Topological data analysis in medical imaging: current state of the art. *Insights into Imaging*, 14(1), 58.

[7] Khalifa, N. E., Loey, M., & Mirjalili, S. (2022). A comprehensive survey of recent trends in deep learning for digital images augmentation. *Artificial Intelligence Review*, 55(3), 2351-2377.

[8] Flanders, A. E., Prevedello, L. M., Shih, G., Halabi, S. S., Kalpathy-Cramer, J., Ball, R., ... & RSNA-ASNR 2019 Brain Hemorrhage CT Annotators. (2020). Radiology: Artificial Intelligence, 2(3), e190211.

[9] Golub, G. H., & Van Loan, C. F. (1996). *Matrix computations*, Johns Hopkins University Press, Baltimore, MD.

[10] Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). *Deep learning* (Vol. 1, No. 2). Cambridge: MIT press.

[11] Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. *The Annals of Mathematical Statistics*, 22(1), 79-86.

[12] De Boer, P. T., Kroese, D. P., Mannor, S., & Rubinstein, R. Y. (2005). A tutorial on the cross-entropy method. *Annals of Operations Research*, 134, 19-67.
