import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class ProtoNet:
    """Prototype Learning for zero-day attack detection"""
    
    def __init__(self, n_prototypes_per_class=10, distance_metric='euclidean'):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.distance_metric = distance_metric
        self.prototypes = None
        self.prototype_labels = None
        self.class_prototypes = {}
        self.distance_thresholds = {}
        
    def fit(self, X, y):
        """Fit prototypes based on training data"""
        print("Training ProtoNet...")
        unique_labels = np.unique(y)
        self.prototypes = []
        self.prototype_labels = []
        
        for label in unique_labels:
            class_data = X[y == label]
            
            if len(class_data) >= self.n_prototypes_per_class:
                # Use K-means to find representative prototypes
                kmeans = KMeans(n_clusters=self.n_prototypes_per_class, random_state=42, n_init=10)
                kmeans.fit(class_data)
                class_prototypes = kmeans.cluster_centers_
            else:
                # If not enough samples, use all samples as prototypes
                class_prototypes = class_data
            
            self.prototypes.extend(class_prototypes)
            self.prototype_labels.extend([label] * len(class_prototypes))
            self.class_prototypes[label] = class_prototypes
            
            # Compute distance thresholds for each class
            if len(class_data) > 0:
                # Calculate distances from prototypes to class data
                distances = []
                for proto in class_prototypes:
                    class_distances = cdist([proto], class_data, metric=self.distance_metric).flatten()
                    distances.extend(class_distances)
                
                # Set threshold as mean + 2*std
                self.distance_thresholds[label] = np.mean(distances) + 2 * np.std(distances)
        
        self.prototypes = np.array(self.prototypes)
        self.prototype_labels = np.array(self.prototype_labels)
        print(f"ProtoNet trained with {len(self.prototypes)} prototypes")
        
    def predict(self, X, threshold_multiplier=2.0):
        """Predict with threshold-based detection for zero-day attacks"""
        predictions = []
        distances_info = []
        
        for x in X:
            # Calculate distances to all prototypes
            distances = cdist([x], self.prototypes, metric=self.distance_metric).flatten()
            min_distance = np.min(distances)
            closest_proto_idx = np.argmin(distances)
            closest_label = self.prototype_labels[closest_proto_idx]
            
            # Get class-specific threshold
            class_threshold = self.distance_thresholds.get(closest_label, np.mean(distances) + threshold_multiplier * np.std(distances))
            
            # Apply threshold multiplier
            adjusted_threshold = class_threshold * threshold_multiplier
            
            # Decision based on distance threshold
            if min_distance <= adjusted_threshold:
                predictions.append(closest_label)
            else:
                predictions.append(-1)  # -1 indicates zero-day attack
            
            distances_info.append({
                'min_distance': min_distance,
                'closest_label': closest_label,
                'threshold': adjusted_threshold
            })
        
        return np.array(predictions), distances_info
    
    def predict_proba(self, X):
        """Get probability-like scores based on distances"""
        proba_scores = []
        
        for x in X:
            distances = cdist([x], self.prototypes, metric=self.distance_metric).flatten()
            # Convert distances to similarity scores (closer = higher score)
            similarity = 1 / (1 + distances)
            normalized_similarity = similarity / np.sum(similarity)
            proba_scores.append(normalized_similarity)
        
        return np.array(proba_scores)
    
    def get_class_distances(self, X):
        """Get distances to each class's prototypes"""
        class_distances = {}
        
        for class_label, prototypes in self.class_prototypes.items():
            # Calculate minimum distance to any prototype in this class
            min_distances = []
            for x in X:
                distances = cdist([x], prototypes, metric=self.distance_metric).flatten()
                min_distances.append(np.min(distances))
            class_distances[class_label] = np.array(min_distances)
        
        return class_distances