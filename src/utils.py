import cv2
import pytesseract
import numpy as np
import re

def calculate_orientation(img):
  # Convert image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Convert image to binary
  _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  # Find all the contours in the thresholded image
  contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv2.contourArea(c)
    # Ignore contours that are too small or too large
    if area < 3700 or 100000 < area:
      continue
    # cv.minAreaRect returns:
    # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # Retrieve the key parameters of the rotated bounding box
    center = (int(rect[0][0]),int(rect[0][1])) 
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = int(rect[2])

    if width < height:
      angle = 90 - angle
    else:
      angle = -angle
    
    return angle


def tesseract_text_orientation(img):
  rotated_image = img
  try:
    output = pytesseract.image_to_osd(img, config='--psm 0')
    angle = re.search(r'Orientation in degrees: \d+', output).group().split(':')[-1].strip()
    if angle == '90':
      rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == '180':
      rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_180)
    if angle == '270':
      rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
    elif angle != 0:    
      rotated_image = calculate_orientation(rotated_image)
  except Exception as e:
    print('error on tesseract_text_orientation:', e)
  finally:
    return rotated_image

def process_rescale(img):
  scale_percent = 220 # percent of original size
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)
  image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  return image

def process_image(image):
  result = image
  result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  result = tesseract_text_orientation(result)
  result = process_rescale(result)
  return result

def process_ocr(img):
  custom_config = r'--oem 3 --psm 5 --psm 6'
  return pytesseract.image_to_string(img, lang='spa', config=custom_config)

def extract_bruto_from_text(text):
  t = text.upper()
  r = '(BRUTO\: *|BRUTO *) *(?P<bruto>\d+\.*\d*)'
  res = re.search(r, t)
  if res is not None:
    t = res.group('bruto')
    bruto = t
    return bruto
  return 0

def extract_neto_from_text(text):
  t = text.upper()
  r = '(NETO\: *|NETO *) *(?P<neto>\d+\.*\d*)'
  res = re.search(r, t)
  if res is not None:
    t = res.group('neto')
    neto = t
    return neto
  return 0

def extract_tara_from_text(text):
  t = text.upper()
  r = '(TARA\: *|TARA *) *(?P<tara>\d+\.*\d*)'
  res = re.search(r, t)
  if res is not None:
    t = res.group('tara')
    tara = t
    return tara
  return 0

def transform_image(img):
    img_resized = img 
    try:
      img_resized = process_image(img_resized)
    except Exception as error:
        print(f'Error on transform image : {error}')
    finally:
      return img_resized