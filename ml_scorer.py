import pandas as pd
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import spacy
import re
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class PureMLStartupScorer:
    def _init_(self, model_path='models/pure_ml_scorer.joblib'):
        self.model_path = model_path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load("en_core_web_sm")
        
        # ML components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)  # For dimensionality reduction
        self.text_clusterer = KMeans(n_clusters=10, random_state=42)
        
        # ML models for each dimension
        self.models = {
            'problem_solution_fit': None,
            'market_potential': None,
            'competitive_advantage': None,
            'feasibility': None
        }
        
        # Training state
        self.is_trained = False
        self.feature_columns = []
        
        # Load pre-trained models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.models = model_data.get('models', self.models)
                self.scaler = model_data.get('scaler', self.scaler)
                self.pca = model_data.get('pca', self.pca)
                self.text_clusterer = model_data.get('text_clusterer', self.text_clusterer)
                self.feature_columns = model_data.get('feature_columns', [])
                self.is_trained = model_data.get('is_trained', False)
                
                if self.is_trained:
                    logging.info("✓ Pure ML models loaded successfully")
                else:
                    logging.info("ℹ Models not trained. Using feature extraction only.")
            else:
                logging.info("ℹ No pre-trained models found. Will extract features only.")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
    
    def _extract_statistical_features(self, text: str) -> np.ndarray:
        """Extract pure statistical features from text"""
        doc = self.nlp(text)
        
        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(list(doc.sents))
        avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Part-of-speech statistics
        pos_counts = {
            'nouns': len([token for token in doc if token.pos_ == 'NOUN']),
            'verbs': len([token for token in doc if token.pos_ == 'VERB']),
            'adjectives': len([token for token in doc if token.pos_ == 'ADJ']),
            'adverbs': len([token for token in doc if token.pos_ == 'ADV']),
            'pronouns': len([token for token in doc if token.pos_ == 'PRON']),
            'conjunctions': len([token for token in doc if token.pos_ == 'CCONJ'])
        }
        
        # Normalize POS counts
        for pos in pos_counts:
            pos_counts[pos] = pos_counts[pos] / word_count if word_count > 0 else 0
        
        # Entity statistics
        entity_counts = {
            'persons': len([ent for ent in doc.ents if ent.label_ == 'PERSON']),
            'organizations': len([ent for ent in doc.ents if ent.label_ == 'ORG']),
            'locations': len([ent for ent in doc.ents if ent.label_ == 'GPE']),
            'products': len([ent for ent in doc.ents if ent.label_ == 'PRODUCT']),
            'money': len([ent for ent in doc.ents if ent.label_ == 'MONEY']),
            'quantities': len([ent for ent in doc.ents if ent.label_ == 'QUANTITY'])
        }
        
        # Normalize entity counts
        for entity in entity_counts:
            entity_counts[entity] = entity_counts[entity] / word_count if word_count > 0 else 0
        
        # Numerical statistics
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        money_amounts = re.findall(r'\$\d+(?:\.\d+)?', text)
        
        numerical_stats = {
            'number_count': len(numbers) / word_count if word_count > 0 else 0,
            'percentage_count': len(percentages) / word_count if word_count > 0 else 0,
            'money_count': len(money_amounts) / word_count if word_count > 0 else 0,
            'avg_number': np.mean([float(n) for n in numbers]) if numbers else 0,
            'has_large_numbers': 1 if any(float(n) > 1000 for n in numbers) else 0
        }
        
        # Readability metrics
        syllable_count = sum(self._count_syllables(word) for word in text.split())
        flesch_score = self._calculate_flesch_score(sentence_count, word_count, syllable_count)
        
        readability_stats = {
            'syllables_per_word': syllable_count / word_count if word_count > 0 else 0,
            'flesch_score': flesch_score,
            'complexity_score': 1 - (flesch_score / 100) if flesch_score > 0 else 0
        }
        
        # Punctuation statistics
        punctuation_stats = {
            'commas': text.count(',') / word_count if word_count > 0 else 0,
            'periods': text.count('.') / word_count if word_count > 0 else 0,
            'question_marks': text.count('?') / word_count if word_count > 0 else 0,
            'exclamation_marks': text.count('!') / word_count if word_count > 0 else 0,
            'colons': text.count(':') / word_count if word_count > 0 else 0,
            'semicolons': text.count(';') / word_count if word_count > 0 else 0
        }
        
        # Combine all features
        all_features = []
        
        # Basic stats
        all_features.extend([word_count, char_count, sentence_count, avg_word_length, avg_sentence_length])
        
        # POS stats
        all_features.extend(list(pos_counts.values()))
        
        # Entity stats
        all_features.extend(list(entity_counts.values()))
        
        # Numerical stats
        all_features.extend(list(numerical_stats.values()))
        
        # Readability stats
        all_features.extend(list(readability_stats.values()))
        
        # Punctuation stats
        all_features.extend(list(punctuation_stats.values()))
        
        return np.array(all_features)
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(syllable_count, 1)
    
    def _calculate_flesch_score(self, sentences: int, words: int, syllables: int) -> float:
        """Calculate Flesch Reading Ease score"""
        if sentences == 0 or words == 0:
            return 0
        
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return max(0, min(100, score))
    
    def _extract_semantic_features(self, text: str) -> np.ndarray:
        """Extract semantic features using embeddings"""
        # Get text embedding
        embedding = self.embedder.encode([text])[0]
        
        # Apply dimensionality reduction if trained
        if self.is_trained:
            try:
                embedding = self.pca.transform([embedding])[0]
            except:
                # If PCA fails, use original embedding
                pass
        
        return embedding
    
    def _extract_contextual_features(self, text: str, competitors: List = None) -> np.ndarray:
        """Extract contextual features from the environment"""
        features = []
        
        # Competitor features
        if competitors:
            features.append(len(competitors))
            features.append(1 if any('http' in str(c) for c in competitors) else 0)
            
            # Average competitor text length
            avg_comp_length = np.mean([len(str(c)) for c in competitors]) if competitors else 0
            features.append(avg_comp_length)
            
            # Competitor embedding similarity
            if self.is_trained:
                text_embedding = self.embedder.encode([text])[0]
                competitor_embeddings = [self.embedder.encode([str(c)])[0] for c in competitors[:5]]
                
                if competitor_embeddings:
                    similarities = [np.dot(text_embedding, comp_emb) / (np.linalg.norm(text_embedding) * np.linalg.norm(comp_emb)) 
                                   for comp_emb in competitor_embeddings]
                    features.append(np.mean(similarities))
                else:
                    features.append(0)
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0])
        
        # Text clustering features
        if self.is_trained:
            try:
                text_embedding = self.embedder.encode([text])[0]
                cluster_label = self.text_clusterer.predict([text_embedding])[0]
                cluster_features = np.zeros(10)
                cluster_features[cluster_label] = 1
                features.extend(cluster_features)
            except:
                features.extend([0] * 10)
        else:
            features.extend([0] * 10)
        
        return np.array(features)
    
    def _extract_all_features(self, idea: str, competitors: List = None, trends_data: str = None) -> np.ndarray:
        """Extract all features without any manual rules"""
        # Statistical features from text
        stat_features = self._extract_statistical_features(idea)
        
        # Semantic features from embeddings
        semantic_features = self._extract_semantic_features(idea)
        
        # Contextual features from environment
        context_features = self._extract_contextual_features(idea, competitors)
        
        # Combine all features
        all_features = np.concatenate([
            stat_features,
            semantic_features,
            context_features
        ])
        
        # Scale features if scaler is trained
        if self.is_trained:
            try:
                all_features = self.scaler.transform([all_features])[0]
            except:
                # If scaling fails, use raw features
                pass
        
        return all_features
    
    def predict(self, idea_description: str, similar_ideas: List = None, trends_data: str = None) -> Dict[str, Any]:
        """
        Predict scores using pure ML approach
        """
        try:
            # Extract features
            features = self._extract_all_features(idea_description, similar_ideas, trends_data)
            
            # Make predictions
            scores = {}
            for dimension, model in self.models.items():
                if model is not None and self.is_trained:
                    try:
                        prediction = model.predict([features])[0]
                        scores[dimension] = np.clip(prediction, 0, 10)
                    except:
                        scores[dimension] = 5.0  # Fallback
                else:
                    scores[dimension] = 5.0  # Fallback
            
            # Calculate overall score
            overall_score = np.mean(list(scores.values()))
            
            # Generate explanations based on feature analysis
            explanations = self._generate_explanations(features, scores, idea_description)
            
            return {
                'scores': scores,
                'overall_score': float(overall_score),
                'explanations': explanations,
                'confidence': 0.85 if self.is_trained else 0.5,
                'scoring_method': 'pure_ml',
                'feature_count': len(features)
            }
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return self._fallback_predict()
    
    def _generate_explanations(self, features: np.ndarray, scores: Dict[str, float], idea: str) -> List[str]:
        """Generate explanations based on feature analysis"""
        explanations = []
        
        # Analyze statistical features (first ~35 features)
        if len(features) > 35:
            stat_features = features[:35]
            
            # Word count analysis
            word_count = stat_features[0]
            if word_count > 20:
                explanations.append("📝 Detailed description provides comprehensive context")
            elif word_count < 10:
                explanations.append("📝 Consider providing more details about your idea")
            
            # Entity analysis
            org_count = stat_features[11]  # Organizations
            money_count = stat_features[15]  # Money mentions
            
            if org_count > 0.05:
                explanations.append("🏢 Business context clearly identified")
            if money_count > 0.02:
                explanations.append("💰 Financial aspects mentioned in description")
            
            # Complexity analysis
            complexity_score = stat_features[32]  # Flesch-based complexity
            if complexity_score > 0.7:
                explanations.append("🧠 Complex concept identified - consider simplification")
            elif complexity_score < 0.3:
                explanations.append("✅ Clear and simple concept presentation")
        
        # Score-based explanations
        for dimension, score in scores.items():
            if score >= 7:
                explanations.append(f"✅ Strong {dimension.replace('_', ' ')} detected")
            elif score < 5:
                explanations.append(f"⚠ {dimension.replace('_', ' ')} needs improvement")
        
        return explanations[:5]  # Limit to top 5
    
    def _fallback_predict(self) -> Dict[str, Any]:
        """Fallback prediction"""
        return {
            'scores': {
                'problem_solution_fit': 5.0,
                'market_potential': 5.0,
                'competitive_advantage': 5.0,
                'feasibility': 5.0,
            },
            'overall_score': 5.0,
            'explanations': ["ML models not trained - using baseline scores"],
            'scoring_method': 'fallback',
            'confidence': 0.3
        }
    
    def train_models(self, training_data_path: str) -> Dict[str, Any]:
        """
        Train ML models on labeled data
        """
        try:
            # Load training data
            df = pd.read_csv(training_data_path)
            
            # Extract features for all training examples
            X = []
            y = {dim: [] for dim in self.models.keys()}
            
            for _, row in df.iterrows():
                # Extract features
                features = self._extract_all_features(
                    row['idea_description'],
                    eval(row.get('similar_ideas', '[]')),
                    row.get('trends_data', '')
                )
                X.append(features)
                
                # Extract target scores
                for dimension in self.models.keys():
                    y[dimension].append(row[dimension])
            
            X = np.array(X)
            
            # Train text clusterer
            text_embeddings = []
            for _, row in df.iterrows():
                embedding = self.embedder.encode([row['idea_description']])[0]
                text_embeddings.append(embedding)
            
            text_embeddings = np.array(text_embeddings)
            self.text_clusterer.fit(text_embeddings)
            
            # Train PCA for dimensionality reduction
            all_embeddings = []
            for _, row in df.iterrows():
                embedding = self.embedder.encode([row['idea_description']])[0]
                all_embeddings.append(embedding)
            
            all_embeddings = np.array(all_embeddings)
            self.pca.fit(all_embeddings)
            
            # Fit scaler
            self.scaler.fit(X)
            
            # Re-extract features with fitted transformers
            X_fitted = []
            for _, row in df.iterrows():
                features = self._extract_all_features(
                    row['idea_description'],
                    eval(row.get('similar_ideas', '[]')),
                    row.get('trends_data', '')
                )
                X_fitted.append(features)
            
            X_fitted = np.array(X_fitted)
            
            # Train models for each dimension
            evaluations = {}
            
            for dimension in self.models.keys():
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_fitted, y[dimension], test_size=0.2, random_state=42
                )
                
                # Train model
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                self.models[dimension] = model
                
                # Evaluate
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                evaluations[dimension] = {
                    'mae': mae,
                    'r2': r2,
                    'samples': len(X_train)
                }
                
                logging.info(f"✓ {dimension} model trained - MAE: {mae:.3f}, R²: {r2:.3f}")
            
            # Save models
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'pca': self.pca,
                'text_clusterer': self.text_clusterer,
                'feature_columns': self.feature_columns,
                'is_trained': True
            }
            
            joblib.dump(model_data, self.model_path)
            self.is_trained = True
            
            logging.info("✓ All pure ML models trained and saved successfully")
            
            return {
                "status": "success",
                "evaluations": evaluations,
                "feature_count": X_fitted.shape[1],
                "training_samples": len(X_fitted),
                "clusters_found": 10
            }
            
        except Exception as e:
            logging.error(f"❌ Training failed: {e}")
            return {"status": "error", "message": str(e)}

# Create global instance
pure_ml_scorer = PureMLStartupScorer()

