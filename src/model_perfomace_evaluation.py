import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluation_model(model, x_test, y_test, output_dir):
    output_dir = str(output_dir)

    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse =np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = pd.DataFrame(
        [
            {
                'MAE' : mae,
                'RMSE' : rmse,
                "R2"  : r2
            }
        ]
    )

    metrics.to_csv(f"{output_dir}/test_metrics.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title("Actual vs Predicted Insurance Charges")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/actual_vs_predicted.png", dpi=150)
    plt.close()

    print("\n===== TEST SET PERFORMANCE =====")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")
    return metrics

