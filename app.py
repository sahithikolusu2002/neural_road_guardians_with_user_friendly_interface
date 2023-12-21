from flask import Flask, render_template, redirect, url_for
import pandas as pd
import plotly.express as px

app = Flask(__name__)


def generate_plots(df, area_name, latitude, longitude, date):
    # Create plots using Plotly Express
    length_of_road = df["X-coordinate"].max() - df["X-coordinate"].min()
    total_potholes = len(df)
    low_potholes = len(df[df["Area_Classification"] == "Low"])
    moderate_potholes = len(df[df["Area_Classification"] == "Medium"])
    high_potholes = len(df[df["Area_Classification"] == "High"])
    highest_depth = df["Actual_Depth"].max()
    severity = df["Depth_Classification"].value_counts()

    fig_length_of_road = px.bar(df, x="Area_Classification", title="Area Severity")
    fig_length_of_road.update_layout(height=400)
    fig_length_of_road.update_layout(
        title=dict(text="Area Severity", font=dict(color="maroon"))
    )

    fig_severity = px.pie(
        severity, values=severity.values, names=severity.index, title="Depth Severity"
    )
    # Add more plots as needed
    fig_severity.update_layout(height=400)
    fig_severity.update_layout(
        title=dict(text="Depth Severity", font=dict(color="maroon"))
    )

    fig_additional_1 = px.scatter(
        df,
        x="X-coordinate",
        y="Y-coordinate",
        title="Geographical Distribution of Potholes",
    )
    # Add additional plots as needed
    fig_additional_1.update_layout(height=300)
    fig_additional_1.update_layout(
        title=dict(
            text="Geographical Distribution of Potholes", font=dict(color="maroon")
        )
    )

    return (
        fig_length_of_road,
        total_potholes,
        low_potholes,
        moderate_potholes,
        high_potholes,
        highest_depth,
        area_name,
        latitude,
        longitude,
        date,
        fig_severity,
        fig_additional_1,
    )


@app.route("/")
def index():
    csv_filename = (
        "C:/Users/kumar/Downloads/Neural_Road_Guardians_Copy/templates/pothole_data.csv"
    )
    try:
        csv_df = pd.read_csv(csv_filename)
        first_row = csv_df.iloc[0]
        latitude = first_row["Latitude"]
        longitude = first_row["Longitude"]
        area_name = first_row["Area"]
        pothole_id = first_row["Pothole ID"]
        date = first_row["Date"]

        (
            fig_length_of_road,
            total_potholes,
            low_potholes,
            moderate_potholes,
            high_potholes,
            highest_depth,
            area_name,
            latitude,
            longitude,
            date,
            fig_severity,
            fig_additional_1,
        ) = generate_plots(csv_df, area_name, latitude, longitude, date)

        if (
            total_potholes == 0
            and low_potholes == 0
            and high_potholes == 0
            and moderate_potholes == 0
        ):
            return redirect(url_for("no_potholes"))

    except pd.errors.EmptyDataError:
        print("CSV file is empty.")
        return redirect(url_for("no_potholes"))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return redirect(url_for("no_potholes"))

    return render_template(
        "dashboard.html",
        fig_length_of_road=fig_length_of_road.to_html(),
        total_potholes=total_potholes,
        low_potholes=low_potholes,
        moderate_potholes=moderate_potholes,
        high_potholes=high_potholes,
        highest_depth=highest_depth,
        pothole_id=pothole_id,
        area_name=area_name,
        latitude=latitude,
        longitude=longitude,
        date=date,
        fig_severity=fig_severity.to_html(),
        fig_additional_1=fig_additional_1.to_html(),
    )


@app.route("/no_potholes")
def no_potholes():
    return render_template("invalid_data.html")


if __name__ == "__main__":
    app.run(debug=True)
