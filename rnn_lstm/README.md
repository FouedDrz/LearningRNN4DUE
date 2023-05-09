# LSTM-Arduino

Ce dépôt contient un exemple de code Arduino pour implémenter un réseau de neurones récurrent à longue mémoire à court terme (LSTM). Il peut être utilisé pour apprendre à partir de données séquentielles et effectuer des prédictions de sortie.

## Fonctionnalités

- Implémentation d'un LSTM avec une couche cachée et une profondeur configurable
- Utilisation des fonctions d'activation sigmoïde et tanh
- Initialisation des poids et des biais en utilisant la méthode de Xavier/Glorot (à effectuer manuellement)

## Prérequis

- Arduino IDE
- Une carte compatible Arduino

## Installation

1. Clonez ce dépôt ou téléchargez le code source en tant que fichier ZIP.
2. Ouvrez le fichier `lstm-rnn.ino` dans l'IDE Arduino.
3. Modifiez les paramètres (taille d'entrée, taille cachée, taille de sortie, profondeur LSTM, taux d'apprentissage) et les données d'entraînement selon vos besoins.
4. Téléversez le code sur votre carte Arduino.

## Utilisation

Le code comprend une fonction `lstm_forward()` pour effectuer une passe avant dans le réseau, et une fonction `rnn_forward()` pour mettre à jour l'état caché et calculer la sortie.

Dans la fonction `setup()`, le LSTM est exécuté avec un exemple d'entrée. Vous pouvez modifier cet exemple d'entrée pour tester le LSTM sur vos propres données. Les valeurs de sortie sont affichées sur le moniteur série.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
