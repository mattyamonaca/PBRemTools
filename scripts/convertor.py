import numpy as np
import pandas as pd
from PIL import Image

def rgb2df(img):
  """
  Convert an RGB image to a DataFrame.
  
  Args:
      img (np.ndarray): RGB image.
      
  Returns:
      df (pd.DataFrame): DataFrame containing the image data.
  """
  h, w, _ = img.shape
  x_l, y_l = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
  r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
  df = pd.DataFrame({
      "x_l": x_l.ravel(),
      "y_l": y_l.ravel(),
      "r": r.ravel(),
      "g": g.ravel(),
      "b": b.ravel(),
  })
  return df

def df2rgba(img_df):
  """
  Convert a DataFrame to an RGB image.
  
  Args:
      img_df (pd.DataFrame): DataFrame containing image data.
      
  Returns:
      img (np.ndarray): RGB image.
  """
  r_img = img_df.pivot_table(index="x_l", columns="y_l",values= "r").reset_index(drop=True).values
  g_img = img_df.pivot_table(index="x_l", columns="y_l",values= "g").reset_index(drop=True).values
  b_img = img_df.pivot_table(index="x_l", columns="y_l",values= "b").reset_index(drop=True).values
  a_img = img_df.pivot_table(index="x_l", columns="y_l",values= "a").reset_index(drop=True).values
  df_img = np.stack([r_img, g_img, b_img, a_img], 2).astype(np.uint8)
  return df_img

def pil2cv(image):
  new_image = np.array(image, dtype=np.uint8)
  if new_image.ndim == 2:
      pass
  elif new_image.shape[2] == 3:
      new_image = new_image[:, :, ::-1]
  elif new_image.shape[2] == 4:
      new_image = new_image[:, :, [2, 1, 0, 3]]
  return new_image

def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image