Smart Waste Management System:-

1. Overview
   This project is a Flask-based web application that classifies waste images into six categories: Plastic, Paper, Cardboard, Glass, Metal, and Trash. It provides disposal suggestions, logs bin fill levels, and uses regression analysis to forecast future capacity.

2. Project Structure
   1. App.py
   2. dataset categories.csv
   3. waste_history.csv
   4. static/
   5. templates/
      - index.html

3. Requirements
   Install the following dependencies in VS Code terminal:
   pip install flask opencv-python scikit-learn matplotlib pandas

4. How to Run
   1. Start the Flask app:
      python App.py
   2. Open your browser and go to:
      http://127.0.0.1:8080
   3. Upload an image (e.g., plastic bottle, newspaper, glass jar).
   4. The app will classify the image and suggest disposal instructions.
   5. View the dashboard:
      - Chart (static/dashboard_chart.png) shows bin capacity forecasts.
      - Logs are stored in waste_history.csv.

5. Features
   1. Waste Classification: Uses filename metadata and KMeans clustering on image pixels.
   2. Disposal Suggestions:
      - Plastic → Recycle in plastic bin
      - Paper → Recycle in paper bin
      - Cardboard → Recycle in cardboard bin
      - Glass → Recycle in glass bin
      - Metal → Recycle in metal bin
      - Trash → Landfill / General Waste
   3. Smart Bin Forecasting:
      - Tracks fill levels in waste_history.csv.
      - Linear regression predicts future capacity.
      - Alerts when bins exceed 85% capacity (“DISPATCH TRUCK”).

6. Example Workflow
   1. Upload plastic_bottle.jpg
   2. App classifies → Plastic
   3. Suggests → "Recycle in plastic bin"
   4. Updates waste_history.csv with new fill level
   5. Dashboard chart updates with forecast

