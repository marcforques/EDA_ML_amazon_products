from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, random_state=42):
    
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    return model

def train_optimized_random_forest(X_train, y_train, best_params, random_state=42):
    
    model = RandomForestClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=-1,
    )
    
    model.fit(X_train, y_train)
    
    return model