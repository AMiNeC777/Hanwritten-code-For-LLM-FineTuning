### Embeddings Notes

1. Où se trouve la structure du réseau (Sequential([Dense, Dropout...])) ?
En PyTorch, l'équivalent est une classe Python qui hérite de nn.Module (juste après la préparation des embeddings).

Voici la correspondance exacte entre le code Keras et le code PyTorch de votre notebook :

Keras / TensorFlow	Équivalent PyTorch dans votre Notebook

Dense(128, activation='relu')	nn.Linear(in_features, 128) suivi de F.relu()

Dropout(0.5)	nn.Dropout(0.5)

Dense(num_classes, activation='softmax')	nn.Linear(hidden_dim, 1) (pour le binaire avec sigmoid/logits)

input_shape=(...)	Géré par la couche d'entrée nn.Embedding puis nn.LSTM
