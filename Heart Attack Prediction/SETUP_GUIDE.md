# Quick Setup Guide
## Heart Attack Risk Prediction - Simplified MLOps System

This guide will help you get the system up and running in minutes.

---

## ⚡ Quick Start (3 Steps)

### Step 1: Install Dependencies

Open your terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

This will install:
- streamlit
- pandas
- numpy
- scikit-learn
- joblib

### Step 2: Configure Dataset

Open `config.json` and verify the target column name matches your dataset:

```json
{
  "training": {
    "dataset_path": "heart_attack_prediction_indonesia.csv",
    "target_column": "target"  ← Make sure this matches your dataset
  }
}
```

**Common target column names**: `target`, `label`, `class`, `heart_attack`, `churn`, `outcome`

### Step 3: Train and Launch

```bash
# Train the initial model
python train.py

# Launch the Streamlit application
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 🔍 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'X'"

**Solution**: Install missing package
```bash
pip install <package-name>
```

Or reinstall all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "Target column 'target' not found"

**Solution**: Update the target column name in `config.json`

1. Open your dataset CSV and identify the actual target column name
2. Edit `config.json`:
   ```json
   "target_column": "your_actual_column_name"
   ```
3. Retry training: `python train.py`

### Issue: "FileNotFoundError: dataset not found"

**Solution**: Verify dataset path

1. Ensure `heart_attack_prediction_indonesia.csv` is in the project directory
2. Or update the path in `config.json`:
   ```json
   "dataset_path": "path/to/your/dataset.csv"
   ```

### Issue: Port 8501 already in use

**Solution**: Use a different port
```bash
streamlit run app.py --server.port 8502
```

---

## 📝 First-Time Checklist

- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset file present in project directory
- [ ] Target column name verified in `config.json`
- [ ] Initial training completed (`python train.py`)
- [ ] Application launched (`streamlit run app.py`)
- [ ] Browser opened to `http://localhost:8501`

---

## 🎯 What to Do After Setup

### 1. Make Your First Prediction
- Navigate to "🔮 Single Prediction"
- Fill in the form with sample data
- Click "Predict Risk"

### 2. Check the Monitoring Dashboard
- Navigate to "📈 Monitoring Dashboard"
- View your training metrics
- Check the performance charts

### 3. Review Model Information
- Navigate to "ℹ️ Model Information"
- Verify model version and hyperparameters
- Check feature list

### 4. Try Batch Prediction
- Prepare a test CSV with the same features as training data
- Navigate to "📊 Batch Prediction"
- Upload and generate predictions

### 5. Explore Retraining
- Navigate to "🔄 Retrain Model"
- Review hyperparameters
- Try adjusting settings and retraining

---

## 📚 Additional Resources

- **Full Documentation**: See `MLOps_Report.md`
- **User Guide**: See `README.md`
- **Implementation Details**: See `walkthrough.md` (in artifacts)

---

## 💡 Tips

1. **Keep backups**: Copy `heart_attack_final_model.pkl` before retraining
2. **Monitor metrics**: Check `metrics_log.csv` to track model evolution
3. **Review logs**: Check `prediction_logs.csv` for prediction history
4. **Experiment safely**: Adjust hyperparameters gradually
5. **Version control**: Commit `config.json` changes to Git

---

## ✅ System Requirements

**Minimum**:
- Python 3.7+
- 2 GB RAM
- 500 MB disk space

**Recommended**:
- Python 3.9+
- 4 GB RAM
- 1 GB disk space

---

## 🚀 Ready to Start!

Once setup is complete, you'll have:
- ✅ A trained machine learning model
- ✅ An interactive web application
- ✅ Monitoring and logging system
- ✅ One-click retraining capability
- ✅ Complete MLOps workflow

**Questions?** Check the documentation files or review the code comments.

**Happy Predicting!** 🎉
