from flask import Flask, request, render_template
from src.Airbnb.pipelines.Prediction_Pipeline import CustomData, PredictPipeline

app = Flask(__name__)


def _get_form_value(field_name: str) -> str:
    value = request.form.get(field_name)
    if value is None or value == "":
        raise ValueError(f"Missing required field: {field_name}")
    return value

# Define the home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Validate and convert form data to CustomData object
            data = CustomData(
                property_type=_get_form_value("property_type"),
                room_type=_get_form_value("room_type"),
                amenities=_get_form_value("amenities"),
                accommodates=_get_form_value("accommodates"),
                bathrooms=_get_form_value("bathrooms"),
                bed_type=_get_form_value("bed_type"),
                cancellation_policy=_get_form_value("cancellation_policy"),
                cleaning_fee=_get_form_value("cleaning_fee"),
                city=_get_form_value("city"),
                host_has_profile_pic=_get_form_value("host_has_profile_pic"),
                host_identity_verified=_get_form_value("host_identity_verified"),
                host_response_rate=_get_form_value("host_response_rate"),
                instant_bookable=_get_form_value("instant_bookable"),
                latitude=_get_form_value("latitude"),
                longitude=_get_form_value("longitude"),
                number_of_reviews=_get_form_value("number_of_reviews"),
                review_scores_rating=_get_form_value("review_scores_rating"),
                bedrooms=_get_form_value("bedrooms"),
                beds=_get_form_value("beds")
            )

            final_data = data.get_data_as_dataframe()

            # Make prediction
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_data)
            result = round(pred[0], 2)
            return render_template("index.html", result=result)

        except Exception as e:
            # Handle exceptions gracefully
            error_message = f"Error during prediction: {str(e)}"
            return render_template("error.html", error_message=error_message)

    else:
        # Render the initial page
        return render_template("index.html", result=None)

# Execution begins
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
