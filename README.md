Smart Waste Management System

Overview:-
This project is a Flask-based web application that classifies waste images into categories (Plastic, Paper, Cardboard, Glass, Metal, Trash) and provides disposal suggestions. It also logs waste fill levels and uses regression to forecast bin capacity.

Project Structure:-
smart-waste-management
│
├── App.py                   -> Flask web app
├── dataset categories.csv   -> Info file describing TrashNet dataset
├── waste_history.csv        -> Auto-generated log of bin fill levels
├── static/                  -> Stores generated charts
├── templates/
│   └── index.html           -> Web interface template

Requirements:-
1.Install dependencies in VS Code terminal:
  pip install flask opencv-python scikit-learn matplotlib pandas
2.How to Run
  Start the Flask app:
   python App.py
3.Open in browser:
   http://127.0.0.1:8080 (127.0.0.1 in Bing)
4.Upload an image:
  > Choose a file (e.g., plastic bottle, newspaper, glass jar).
      > The app will classify it and suggest disposal instructions.
5.View dashboard:
  The app generates a chart (static/dashboard_chart.png) showing bin capacity forecasts
6.Waste logs are stored in waste_history.csv.

Features:-
 Waste Classification:
 Uses filename metadata + KMeans clustering on image pixels.
 Categories: Plastic, Paper, Cardboard, Glass, Metal, Trash.
 
 Disposal Suggestions:
Plastic   -> Recycle in plastic bin
Paper     -> Recycle in paper bin
Cardboard -> Recycle in cardboard bin
Glass     -> Recycle in glass bin
Metal     -> Recycle in metal bin
Trash     -> Landfill / General Waste

Smart Bin Forecasting:
Logs fill levels in waste_history.csv.
Linear regression predicts future capacity.
Alerts when bins exceed 85% capacity (“DISPATCH TRUCK”).

Example Workflow
Upload plastic_bottle.jpg
App classifies -> Plastic
Suggests -> "Recycle in plastic bin"
Updates waste_history.csv with new fill level
Dashboard chart updates with forecast

.

Logs auto-reset if you press "Reset Database" on the web page.
