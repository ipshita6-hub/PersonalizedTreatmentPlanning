#!/usr/bin/env python3
"""
Quick Model Performance Improvement Script.
Focuses on the most effective improvements for faster results.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor, generate_sample_data
from enhanced_feature_engineering import EnhancedFeatureEngineer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)


def quick_hyperparameter_tuning(X_train, y_train, model_type='RandomForest'):
    """Quick hyperparameter tuning with focused parameter grid."""
    print(f"Quick tuning for {model_type}...")
    
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
    else:
        return None, 0
    
    # Quick grid search with fewer folds
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='accuracy', 
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_score_


def main():
    """Main function for quick model improvement."""
    print("⚡ QUICK MODEL PERFORMANCE IMPROVEMENT")
    print("=" * 60)
    print("Fast-track improvements focusing on the most effective techniques")
    
    results = {}
    
    try:
        # Step 1: Generate data
        print_section("STEP 1: DATA PREPARATION")
        
        print("Generating dataset (3000 samples)...")
        sample_data = generate_sample_data(3000)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.clean_data(sample_data)
        
        categorical_cols = ['gender', 'symptoms', 'diagnosis', 'previous_treatment', 'severity']
        encoded_data = preprocessor.encode_categorical_features(clean_data, categorical_cols)
        
        print(f"✓ Dataset prepared: {encoded_data.shape}")
        
        # Step 2: Baseline Performance
        print_section("STEP 2: BASELINE MODEL")
        
        from train_model import OutcomePredictionTrainer
        trainer = OutcomePredictionTrainer()
        X_basic, y = trainer.prepare_outcome_data(encoded_data)
        
        # Basic features only
        basic_features = ['age', 'gender', 'symptoms', 'diagnosis', 'previous_treatment', 'severity']
        X_baseline = X_basic[basic_features]
        
        # Split and scale
        X_train_base, X_test_base, y_train, y_test = preprocessor.split_data(X_baseline, y, test_size=0.2, random_state=42)
        X_train_base_scaled, X_test_base_scaled = preprocessor.scale_features(X_train_base, X_test_base)
        
        # Baseline model
        baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        baseline_cv = cross_val_score(baseline_model, X_train_base_scaled, y_train, cv=5, scoring='accuracy')
        baseline_score = baseline_cv.mean()
        
        baseline_model.fit(X_train_base_scaled, y_train)
        baseline_test = baseline_model.score(X_test_base_scaled, y_test)
        
        print(f"✓ Baseline CV: {baseline_score:.4f} (+/- {baseline_cv.std()*2:.4f})")
        print(f"✓ Baseline Test: {baseline_test:.4f}")
        
        results['baseline'] = {'cv': baseline_score, 'test': baseline_test, 'features': len(basic_features)}
        
        # Step 3: Enhanced Features
        print_section("STEP 3: ENHANCED FEATURE ENGINEERING")
        
        enhanced_engineer = EnhancedFeatureEngineer()
        enhanced_data = enhanced_engineer.engineer_all_features(encoded_data, include_polynomial=False, include_clustering=False)
        
        # Prepare enhanced features
        X_enhanced, y_enh = trainer.prepare_outcome_data(enhanced_data)
        X_enhanced_numeric = X_enhanced.select_dtypes(include=[np.number])
        X_enhanced_clean = X_enhanced_numeric.fillna(X_enhanced_numeric.median())
        
        print(f"✓ Enhanced features: {X_enhanced_clean.shape[1]} (added {X_enhanced_clean.shape[1] - len(basic_features)})")
        
        # Split enhanced data
        X_train_enh, X_test_enh, y_train_enh, y_test_enh = preprocessor.split_data(X_enhanced_clean, y_enh, test_size=0.2, random_state=42)
        X_train_enh_scaled, X_test_enh_scaled = preprocessor.scale_features(X_train_enh, X_test_enh)
        
        # Test enhanced features
        enhanced_model = RandomForestClassifier(n_estimators=100, random_state=42)
        enhanced_cv = cross_val_score(enhanced_model, X_train_enh_scaled, y_train_enh, cv=5, scoring='accuracy')
        enhanced_score = enhanced_cv.mean()
        
        enhanced_model.fit(X_train_enh_scaled, y_train_enh)
        enhanced_test = enhanced_model.score(X_test_enh_scaled, y_test_enh)
        
        print(f"✓ Enhanced CV: {enhanced_score:.4f} (+/- {enhanced_cv.std()*2:.4f})")
        print(f"✓ Enhanced Test: {enhanced_test:.4f}")
        
        improvement_1 = enhanced_score - baseline_score
        print(f"✓ Improvement: {improvement_1:+.4f} ({improvement_1/baseline_score*100:+.1f}%)")
        
        results['enhanced'] = {'cv': enhanced_score, 'test': enhanced_test, 'features': X_enhanced_clean.shape[1]}
        
        # Step 4: Feature Selection
        print_section("STEP 4: FEATURE SELECTION")
        
        # Test different numbers of features
        best_k = 10
        best_score = 0
        
        for k in [8, 10, 12, 15, 20]:
            if k >= X_enhanced_clean.shape[1]:
                continue
                
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X_train_enh_scaled, y_train_enh)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            cv_scores = cross_val_score(model, X_selected, y_train_enh, cv=3, scoring='accuracy')
            score = cv_scores.mean()
            
            print(f"  Top {k} features: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"✓ Best feature count: {best_k} features ({best_score:.4f})")
        
        # Apply best feature selection
        selector = SelectKBest(score_func=f_classif, k=best_k)
        X_train_selected = selector.fit_transform(X_train_enh_scaled, y_train_enh)
        X_test_selected = selector.transform(X_test_enh_scaled)
        
        # Test selected features
        selected_model = RandomForestClassifier(n_estimators=100, random_state=42)
        selected_cv = cross_val_score(selected_model, X_train_selected, y_train_enh, cv=5, scoring='accuracy')
        selected_score = selected_cv.mean()
        
        selected_model.fit(X_train_selected, y_train_enh)
        selected_test = selected_model.score(X_test_selected, y_test_enh)
        
        print(f"✓ Selected Features CV: {selected_score:.4f}")
        print(f"✓ Selected Features Test: {selected_test:.4f}")
        
        improvement_2 = selected_score - baseline_score
        print(f"✓ Total Improvement: {improvement_2:+.4f} ({improvement_2/baseline_score*100:+.1f}%)")
        
        results['selected'] = {'cv': selected_score, 'test': selected_test, 'features': best_k}
        
        # Step 5: Quick Hyperparameter Tuning
        print_section("STEP 5: HYPERPARAMETER TUNING")
        
        # Tune Random Forest
        rf_tuned, rf_score = quick_hyperparameter_tuning(X_train_selected, y_train_enh, 'RandomForest')
        
        # Tune Gradient Boosting
        gb_tuned, gb_score = quick_hyperparameter_tuning(X_train_selected, y_train_enh, 'GradientBoosting')
        
        # Choose best model
        if gb_score > rf_score:
            best_tuned_model = gb_tuned
            best_tuned_score = gb_score
            best_model_name = "Gradient Boosting"
        else:
            best_tuned_model = rf_tuned
            best_tuned_score = rf_score
            best_model_name = "Random Forest"
        
        # Test best tuned model
        tuned_test = best_tuned_model.score(X_test_selected, y_test_enh)
        
        print(f"✓ Best tuned model: {best_model_name}")
        print(f"✓ Tuned CV: {best_tuned_score:.4f}")
        print(f"✓ Tuned Test: {tuned_test:.4f}")
        
        final_improvement = best_tuned_score - baseline_score
        print(f"✓ Final Improvement: {final_improvement:+.4f} ({final_improvement/baseline_score*100:+.1f}%)")
        
        results['tuned'] = {'cv': best_tuned_score, 'test': tuned_test, 'features': best_k}
        
        # Step 6: Final Evaluation
        print_section("STEP 6: FINAL RESULTS")
        
        print("📊 PERFORMANCE SUMMARY:")
        print(f"   Baseline:           {baseline_score:.4f} ({len(basic_features)} features)")
        print(f"   Enhanced Features:  {enhanced_score:.4f} ({X_enhanced_clean.shape[1]} features)")
        print(f"   Feature Selection:  {selected_score:.4f} ({best_k} features)")
        print(f"   Hyperparameter Tuning: {best_tuned_score:.4f} ({best_k} features)")
        
        print(f"\n🎯 IMPROVEMENT BREAKDOWN:")
        print(f"   Enhanced Features:  {improvement_1:+.4f} ({improvement_1/baseline_score*100:+.1f}%)")
        print(f"   Feature Selection:  {(selected_score-enhanced_score):+.4f} ({(selected_score-enhanced_score)/baseline_score*100:+.1f}%)")
        print(f"   Hyperparameter Tuning: {(best_tuned_score-selected_score):+.4f} ({(best_tuned_score-selected_score)/baseline_score*100:+.1f}%)")
        print(f"   TOTAL IMPROVEMENT:  {final_improvement:+.4f} ({final_improvement/baseline_score*100:+.1f}%)")
        
        # Save improved model
        print(f"\n💾 SAVING IMPROVED MODEL:")
        
        import pickle
        os.makedirs('models', exist_ok=True)
        
        with open('models/quick_improved_model.pkl', 'wb') as f:
            pickle.dump(best_tuned_model, f)
        
        with open('models/quick_feature_selector.pkl', 'wb') as f:
            pickle.dump(selector, f)
        
        with open('models/quick_preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        
        # Save selected feature names
        selected_feature_names = X_enhanced_clean.columns[selector.get_support()].tolist()
        import json
        with open('models/quick_selected_features.json', 'w') as f:
            json.dump(selected_feature_names, f)
        
        print("   ✓ Model saved to models/quick_improved_model.pkl")
        print("   ✓ Feature selector saved")
        print("   ✓ Preprocessor saved")
        
        # Performance assessment
        if final_improvement/baseline_score > 0.15:
            print(f"\n🎉 EXCELLENT IMPROVEMENT! (+{final_improvement/baseline_score*100:.1f}%)")
            print("   Your model is significantly better and ready for deployment!")
        elif final_improvement/baseline_score > 0.05:
            print(f"\n✅ GOOD IMPROVEMENT! (+{final_improvement/baseline_score*100:.1f}%)")
            print("   Solid progress - ready for web interface!")
        else:
            print(f"\n👍 MODERATE IMPROVEMENT (+{final_improvement/baseline_score*100:.1f}%)")
            print("   Some progress made - consider more advanced techniques")
        
        print(f"\n🚀 READY FOR OPTION A: WEB INTERFACE!")
        print("   Your improved model is now ready for web deployment")
        
        return {
            'baseline_score': baseline_score,
            'final_score': best_tuned_score,
            'improvement': final_improvement,
            'improvement_pct': final_improvement/baseline_score*100,
            'best_model': best_tuned_model,
            'selector': selector,
            'preprocessor': preprocessor,
            'results': results
        }
        
    except Exception as e:
        print(f"\n❌ Error during improvement: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\n🎯 SUCCESS! Model improved by {results['improvement_pct']:.1f}%")
        print(f"   From {results['baseline_score']:.4f} to {results['final_score']:.4f}")
    else:
        print(f"\n❌ Improvement failed. Please check errors above.")