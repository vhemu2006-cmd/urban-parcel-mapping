# 🛰️ Urban Parcel Mapping using AI

An AI-powered remote sensing application that analyzes urban satellite images and identifies different land-use categories such as buildings, roads, water, forests, barren land, and agriculture.

The system uses a deep learning semantic segmentation model to generate a land-use map and calculate the percentage distribution of each category.

---

## 🚀 Project Overview

Urban planning and land-use analysis are important for smart city development, infrastructure planning, environmental monitoring, and disaster management.

Traditional land-use mapping requires significant manual effort. This project automates the process using satellite imagery and deep learning.

The application accepts an urban image as input and produces:

- Original satellite image
- Predicted segmentation map
- Land-use classification
- Pixel-wise land-use percentages
- AI-generated land-use analysis report

---

## ✨ Features

- Upload satellite or remote sensing images
- AI-based semantic segmentation
- Detects multiple urban land-use classes
- Generates colored prediction maps
- Calculates land-use percentages
- Displays original and predicted images
- Provides AI-based interpretation
- Interactive web interface
- FastAPI backend
- U-Net deep learning model

---

## 🧠 Land-Use Classes

The model identifies the following classes:

| Class ID | Land-Use Category |
|----------|-------------------|
| 0 | No-data |
| 1 | Background |
| 2 | Building |
| 3 | Road |
| 4 | Water |
| 5 | Barren Land |
| 6 | Forest |
| 7 | Agriculture |

---

## 🏗️ System Architecture

```text
Satellite Image
      |
      v
Frontend Image Upload
      |
      v
FastAPI Backend
      |
      v
Image Preprocessing
      |
      v
U-Net + ResNet18 Model
      |
      v
Semantic Segmentation
      |
      v
Land-Use Percentage Calculation
      |
      v
AI Analysis Report
      |
      v
Results Displayed on Web Interface
