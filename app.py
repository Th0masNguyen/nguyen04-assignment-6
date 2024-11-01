from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):
    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)  # X values between 0 and 1
    sigma = np.sqrt(sigma2)  # Standard deviation from variance
    Y = mu + sigma * np.random.randn(N)  # Y values with normal distribution

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]  # Extract slope from the fitted model
    intercept = model.intercept_  # Extract intercept from the fitted model

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y, label='Data Points')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', label='Regression Line')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Regression Line: Y = {intercept:.2f} + {slope:.2f}X")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations and create histograms of slopes and intercepts
    slopes = []  # Initialize empty list for slopes
    intercepts = []  # Initialize empty list for intercepts

    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = mu + sigma * np.random.randn(N)

        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)

        slopes.append(sim_model.coef_[0])  # Append slope
        intercepts.append(sim_model.intercept_)  # Append intercept

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of more extreme slopes and intercepts
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        # Return the rendered template with input values
        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme,
                               N=N, mu=mu, sigma2=sigma2, S=S)

    # Render the template for the first time with empty inputs
    return render_template("index.html", N="", mu="", sigma2="", S="")

if __name__ == "__main__":
    app.run(debug=True)
