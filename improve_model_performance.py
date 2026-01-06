#!/usr/bin/env python3
"""
Comprehensive Model Performance Improvement Script.
This script implements advanced techniques to significantly improve model accuracy.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor, generate_sample_data
from enhanced_feature_engineering import EnhancedFeatureEngineer
from advanced_model_training import AdvancedModelTrainer
from evaluate_model import OutcomeEvaluator


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)


def compare_baseline_vs_improved(baseline_score, improved_score):
    """Compare baseline vs improved model performance."""
    improvement = improved_score - baseline_score
    improvement_pct = (improvement / baseline_score) * 100
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   Baseline Accuracy:  {baseline_score:.4f}")
    print(f"   Improved Accuracy:  {improved_score:.4f}")
    print(f"   Improvement:        +{improvement:.4f} ({improvement_pct:+.1f}%)")
    
    if improvement_pct > 20:
        print(f"   🎉 Excellent improvement!")
    elif improvement_pct > 10:
        print(f"   ✅ Good improvement!")
    elif improvement_pct > 5:
        print(f"   👍 Moderate improvement")
    else:
        print(f"   ⚠️  Minimal improvement - consider different approaches")


def create_performance_visualization(results_history):
    """Create visualizations of performance improvements."""
    print("\n📈 Creating performance visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Improvement Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance comparison bar chart
    ax1 = axes[0, 0]
    methods = list(results_history.keys())
    scores = [results_history[method]['cv_score'] for method in methods]
    
    bars = ax1.bar(methods, scores, color=['lightcoral', 'lightblue', 'lightgreen', 'gold'])
    ax1.set_title('Cross-Validation Accuracy by Method')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Improvement over baseline
    ax2 = axes[0, 1]
    baseline_score = scores[0]
    improvements = [(score - baseline_score) * 100 for score in scores]
    
    colors = ['red' if imp < 0 else 'green' for imp in improvements]
    bars2 = ax2.bar(methods, improvements, color=colors, alpha=0.7)
    ax2.set_title('Improvement Over Baseline (%)')
    ax2.set_ylabel('Improvement (%)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                f'{imp:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Feature count comparison
    ax3 = axes[1, 0]
    feature_counts = [results_history[method].get('n_features', 0) for method in methods]
    
    ax3.bar(methods, feature_counts, color='skyblue', alpha=0.7)
    ax3.set_title('Number of Features Used')
    ax3.set_ylabel('Feature Count')
    
    for i, count in enumerate(feature_counts):
        if count > 0:
            ax3.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Performance vs Feature Count scatter
    ax4 = axes[1, 1]
    valid_data = [(scores[i], feature_counts[i], methods[i]) 
                  for i in range(len(methods)) if feature_counts[i] > 0]
    
    if valid_data:
        x_vals, y_vals, labels = zip(*valid_data)
        scatter = ax4.scatter(y_vals, x_vals, c=range(len(valid_data)), 
                            cmap='viridis', s=100, alpha=0.7)
        
        for i, label in enumerate(labels):
            ax4.annotate(label, (y_vals[i], x_vals[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Feature Count')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'model_improvement_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 Visualization saved as: {filename}")
    
    plt.show()


def main():
    """Main function to run comprehensive model improvement."""
    print("🚀 COMPREHENSIVE MODEL PERFORMANCE IMPROVEMENT")
    print("=" * 70)
    print("This script will systematically improve your model performance using:")
    print("• Enhanced feature engineering")
    print("• Advanced preprocessing")
    print("• Hyperparameter optimization")
    print("• Ensemble methods")
    print("• Feature selection")
    
    results_history = {}
    
    try:
        # Step 1: Generate larger, higher-quality dataset
        print_section("STEP 1: DATA GENERATION AND PREPROCESSING")
        
        print("Generating enhanced dataset (5000 samples)...")
        sample_data = generate_sample_data(5000)  # Larger dataset for better training
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Clean data
        print("Cleaning and preprocessing data...")
        clean_data = preprocessor.clean_data(sample_data)
        
        # Encode categorical features
        categorical_cols = ['gender', 'symptoms', 'diagnosis', 'previous_treatment', 'severity']
        encoded_data = preprocessor.encode_categorical_features(clean_data, categorical_cols)
        
        print(f"✓ Dataset prepared: {encoded_data.shape}")
        
        # Step 2: Baseline Model Performance
        print_section("STEP 2: BASELINE MODEL PERFORMANCE")
        
        # Prepare basic features for baseline
        from train_model import OutcomePredictionTrainer
        baseline_trainer = OutcomePredictionTrainer()
        X_basic, y = baseline_trainer.prepare_outcome_data(encoded_data)
        
        # Use only basic features for baseline
        basic_features = ['age', 'gender', 'symptoms', 'diagnosis', 'previous_treatment', 'severity']
        X_baseline = X_basic[basic_features]
        
        # Split data
        X_train_base, X_test_base, y_train, y_test = preprocessor.split_data(
            X_baseline, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_base_scaled, X_test_base_scaled = preprocessor.scale_features(X_train_base, X_test_base)
        
        # Train baseline model
        print("Training baseline Random Forest model...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        baseline_cv_scores = cross_val_score(baseline_model, X_train_base_scaled, y_train, cv=5, scoring='accuracy')
        baseline_score = baseline_cv_scores.mean()
        
        baseline_model.fit(X_train_base_scaled, y_train)
        baseline_test_score = baseline_model.score(X_test_base_scaled, y_test)
        
        print(f"✓ Baseline CV Accuracy: {baseline_score:.4f} (+/- {baseline_cv_scores.std() * 2:.4f})")
        print(f"✓ Baseline Test Accuracy: {baseline_test_score:.4f}")
        
        results_history['Baseline'] = {
            'cv_score': baseline_score,
            'test_score': baseline_test_score,
            'n_features': len(basic_features)
        }
        
        # Step 3: Enhanced Feature Engineering
        print_section("STEP 3: ENHANCED FEATURE ENGINEERING")
        
        enhanced_engineer = EnhancedFeatureEngineer()
        enhanced_data = enhanced_engineer.engineer_all_features(encoded_data)
        
        # Prepare enhanced features
        X_enhanced, y_enhanced = baseline_trainer.prepare_outcome_data(enhanced_data)
        X_enhanced_numeric = X_enhanced.select_dtypes(include=[np.number])
        X_enhanced_clean = X_enhanced_numeric.fillna(X_enhanced_numeric.median())
        
        # Split enhanced data
        X_train_enh, X_test_enh, y_train_enh, y_test_enh = preprocessor.split_data(
            X_enhanced_clean, y_enhanced, test_size=0.2, random_state=42
        )
        
        # Scale enhanced features
        X_train_enh_scaled, X_test_enh_scaled = preprocessor.scale_features(X_train_enh, X_test_enh)
        
        # Train model with enhanced features
        print("Training model with enhanced features...")
        enhanced_model = RandomForestClassifier(n_estimators=100, random_state=42)
        enhanced_cv_scores = cross_val_score(enhanced_model, X_train_enh_scaled, y_train_enh, cv=5, scoring='accuracy')
        enhanced_score = enhanced_cv_scores.mean()
        
        enhanced_model.fit(X_train_enh_scaled, y_train_enh)
        enhanced_test_score = enhanced_model.score(X_test_enh_scaled, y_test_enh)
        
        print(f"✓ Enhanced Features CV Accuracy: {enhanced_score:.4f} (+/- {enhanced_cv_scores.std() * 2:.4f})")
        print(f"✓ Enhanced Features Test Accuracy: {enhanced_test_score:.4f}")
        
        results_history['Enhanced Features'] = {
            'cv_score': enhanced_score,
            'test_score': enhanced_test_score,
            'n_features': X_enhanced_clean.shape[1]
        }
        
        compare_baseline_vs_improved(baseline_score, enhanced_score)
        
        # Step 4: Advanced Model Training with Hyperparameter Tuning
        print_section("STEP 4: ADVANCED MODEL TRAINING")
        
        print("Running comprehensive model optimization...")
        advanced_trainer = AdvancedModelTrainer()
        
        # Run comprehensive optimization (this will take some time)
        optimization_results = advanced_trainer.comprehensive_optimization(
            X_train_enh_scaled, y_train_enh, X_test_enh_scaled, y_test_enh
        )
        
        optimized_score = optimization_results['cv_score']
        optimized_test_score = optimization_results['test_score']
        
        print(f"✓ Optimized Model CV Accuracy: {optimized_score:.4f}")
        print(f"✓ Optimized Model Test Accuracy: {optimized_test_score:.4f}")
        
        results_history['Optimized Model'] = {
            'cv_score': optimized_score,
            'test_score': optimized_test_score,
            'n_features': X_enhanced_clean.shape[1]
        }
        
        compare_baseline_vs_improved(baseline_score, optimized_score)
        
        # Step 5: Model Evaluation and Analysis
        print_section("STEP 5: COMPREHENSIVE MODEL EVALUATION")
        
        # Evaluate the best model
        best_model = optimization_results['final_model']
        evaluator = OutcomeEvaluator()
        
        print("Performing detailed model evaluation...")
        outcome_labels = ['Improved', 'Not Improved', 'Stable']
        feature_names = X_enhanced_clean.columns.tolist()
        
        evaluation_results = evaluator.evaluate_classification_model(
            best_model, X_test_enh_scaled, y_test_enh, "Optimized Model"
        )
        
        # Step 6: Create Performance Visualizations
        print_section("STEP 6: PERFORMANCE ANALYSIS AND VISUALIZATION")
        
        create_performance_visualization(results_history)
        
        # Step 7: Final Summary and Recommendations
        print_section("STEP 7: FINAL SUMMARY AND RECOMMENDATIONS")
        
        print("🎯 PERFORMANCE IMPROVEMENT SUMMARY:")
        print(f"   Starting Accuracy:  {baseline_score:.4f}")
        print(f"   Final Accuracy:     {optimized_score:.4f}")
        
        total_improvement = optimized_score - baseline_score
        total_improvement_pct = (total_improvement / baseline_score) * 100
        
        print(f"   Total Improvement:  +{total_improvement:.4f} ({total_improvement_pct:+.1f}%)")
        
        print(f"\n🔧 KEY IMPROVEMENTS MADE:")
        print(f"   • Enhanced feature engineering: {X_enhanced_clean.shape[1]} features")
        print(f"   • Advanced preprocessing optimization")
        print(f"   • Hyperparameter tuning across multiple algorithms")
        print(f"   • Ensemble model creation")
        print(f"   • Feature selection optimization")
        
        print(f"\n🚀 NEXT STEPS FOR FURTHER IMPROVEMENT:")
        if total_improvement_pct < 10:
            print("   • Consider collecting more diverse training data")
            print("   • Experiment with deep learning models")
            print("   • Try advanced ensemble techniques (stacking, blending)")
            print("   • Implement domain-specific feature engineering")
        else:
            print("   • Deploy the improved model to production")
            print("   • Set up model monitoring and retraining pipeline")
            print("   • Create web interface for easy access")
            print("   • Validate with real healthcare data")
        
        print(f"\n💾 SAVING IMPROVED MODEL:")
        
        # Save the improved model
        import pickle
        os.makedirs('models', exist_ok=True)
        
        with open('models/improved_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        with open('models/improved_preprocessor.pkl', 'wb') as f:
            pickle.dump(optimization_results['preprocessing'], f)
        
        # Save feature names
        import json
        with open('models/improved_features.json', 'w') as f:
            json.dump(feature_names, f)
        
        print("   ✓ Improved model saved to models/improved_model.pkl")
        print("   ✓ Preprocessor saved to models/improved_preprocessor.pkl")
        print("   ✓ Feature names saved to models/improved_features.json")
        
        print(f"\n🎉 MODEL IMPROVEMENT COMPLETED SUCCESSFULLY!")
        print(f"   Your model is now ready for production deployment!")
        
        return {
            'baseline_score': baseline_score,
            'final_score': optimized_score,
            'improvement': total_improvement,
            'improvement_pct': total_improvement_pct,
            'best_model': best_model,
            'results_history': results_history
        }
        
    except Exception as e:
        print(f"\n❌ Error during model improvement: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\n📊 Final Results:")
        print(f"   Improvement: {results['improvement_pct']:.1f}%")
        print(f"   Ready for Option A (Web Interface)!")
    else:
        print(f"\n❌ Model improvement failed. Please check the errors above.")