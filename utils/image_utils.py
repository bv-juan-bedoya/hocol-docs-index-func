import fitz  # PyMuPDF
import os
import cv2
import numpy as np
import pytesseract
import logging


def detect_and_crop_tables(image):
    detection_img = image.copy()
    gray = cv2.cvtColor(detection_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 150 <= h <= 1500 and w >= 1200:
            valid_tables.append((x, y, w, h))
    return valid_tables


# Define the function for detecting tables on the first cropped image
def detect_first_page_table(image):
    detection_img = image.copy()
    gray = cv2.cvtColor(detection_img, cv2.COLOR_BGR2GRAY)

    # Umbral para inversión binaria
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Kernel adaptado para tablas horizontales con estructura rectangular
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Detección de contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_table = None
    image_height = image.shape[0]

    logging.info(f"Found {len(contours)} contours in first page image")
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtrar por posición vertical (solo parte inferior de la imagen)
        is_low = y > image_height * 0.5  # Parte baja de la imagen

        # Filtrar por tamaño aproximado esperado de la tabla
        if is_low and h > 100 and w > 1000:
            target_table = (x, y, w, h)
            logging.info(f"Found target table: x={x}, y={y}, w={w}, h={h}")
            break  # Solo la primera tabla válida encontrada en la parte baja

    return target_table