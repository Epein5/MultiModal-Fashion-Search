# Fashion Image Prediction System

This project utilizes the CLIP(Contrastive Language-Image Pre-Training) model to predict fashion images from text descriptions. It offers image embedding prediction for finding similar products and features a user-friendly interface for easy searching and browsing, streamlining fashion product discovery and recommendations.

## Project Structure
    ├── Dataset/
    │   ├── images/
    │   └── styles/
    ├── Models/
    ├── pickels/
    ├── Code/
    │   ├── 44kpreprocessing.ipynb
    │   ├── train.ipynb
    │   └── evaluation.ipynb
    ├── app/
    │   ├── classes.py
    │   └── main.py
    ├── requirements.txt
    └── README.md

## Setup and Installation

1. Clone the repository:

       git clone https://github.com/your-username/fashion-image-prediction.git
       cd fashion-image-prediction

2. Create and set up the dataset:
  - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
  - Create a `Dataset` directory and place the `images` and `styles` folders inside it

3. Create directories for models and pickles:

4. Install dependencies:

       pip install -r requirements.txt

5. Preprocess the data:
   - Run `Code/44kpreprocessing.ipynb` to create a CSV file with the following structure:
     ```
     image,        caption,                                                          caption_number,  id
  
      15970.jpg,    Checked Navy Blue Shirts for Men with Long Sleeves,                0,              0
      53759.jpg,    Solid Grey Tshirts for Men with Short Sleeves and Polo Collar,     0,              1
      1855.jpg,     Printed Grey Tshirts for Men with Short Sleeves and Round Neck,    0,              2
      30805.jpg,    Striped Green Shirts for Men with Long Sleeves,                    0,              3
      26960.jpg,    Solid Purple Shirts for Women with Short Sleeves,                  0,              4
     ```

   **Note:** If you have multiple captions for one image, use the following format for `captions.csv`:
     ```
     image,        caption,                                                          caption_number,  id
     15970.jpg,    Caption no 1,                                                     0,               0
     15970.jpg,    Caption no 2,                                                     1,               0
     ```
   Ensure that all images have the same number of captions.

6. Train the model:
   - Run `Code/train.ipynb`
   - Adjust the paths for saving Models and pickels as needed
   - If using multiple captions per image, set `num_workers` in the `CFG` class to 1
  The training process will generate a graph showing the validation and training loss:
  
![loss (1)](https://github.com/Epein5/MultiModal-Fashion-Search/assets/110723354/e6395f5d-e2ac-48da-80b6-795c8058578b)

   This graph illustrates the model's performance over time, with the blue line representing training loss and the orange line representing validation loss.


7. Evaluate the model (optional):
- Run `Code/evaluation.ipynb`

## Running the Web Application

1. Adjust paths in `app/classes.py` as required

2. Run the following command from the root directory:

       uvicorn app.main:app
## Evaluation Results

The model evaluation yielded an average similarity score of 0.8041769867063955 between original captions and generated captions. This method was preferred over directly comparing text features and image embeddings, which resulted in a lower mean similarity of 0.015416127629578114.

## Results

### Final Output
Our model produces high-quality image predictions based on textual input. Here are examples of our results:

<p float="left">
  <img src="https://github.com/Epein5/MultiModal-Fashion-Search/assets/110723354/ab84d7ad-5b57-4791-9316-77b8ee91e010" width="45%" alt="Front View Example 1" />
  <img src="https://github.com/Epein5/MultiModal-Fashion-Search/assets/110723354/3675ab10-54a8-4d2b-a597-20f6bae88b54" width="45%" alt="Side View Example 1" />
</p>

*These images showcase our model's ability to predict and match fashion items based on textual descriptions.*

### Video Demonstration

For a more detailed look at how our system works, check out this video demonstration:

[![Fashion Image Prediction Demo](https://img.youtube.com/vi/cIMYi6iZdOE/0.jpg)](https://youtu.be/cIMYi6iZdOE)

*Click the image above to watch the video on YouTube*
## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.
