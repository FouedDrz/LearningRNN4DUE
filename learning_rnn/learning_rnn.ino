#include <Arduino.h>

// Paramètres
const int INPUT_SIZE = 3; // Exemple de taille d'entrée
const int HIDDEN_SIZE = 4; // Exemple de taille cachée
const int OUTPUT_SIZE = 2; // Exemple de taille de sortie (nombre de classes)
const float LR = 0.01; // Taux d'apprentissage

// Poids et biais du modèle
float Wxh[HIDDEN_SIZE][INPUT_SIZE]; // Poids de l'entrée à la couche cachée
float Whh[HIDDEN_SIZE][HIDDEN_SIZE]; // Poids de la couche cachée à la couche cachée
float bh[HIDDEN_SIZE]; // Biais de la couche cachée
float Why[OUTPUT_SIZE][HIDDEN_SIZE]; // Poids de la couche cachée à la couche de sortie
float by[OUTPUT_SIZE]; // Biais de la couche de sortie

// Fonction d'activation
float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

float dsigmoid(float y) {
  return y * (1 - y);
}
//

void initialize_weights_biases() {
  // Initialiser les poids avec la méthode de Xavier/Glorot
  float Wxh_scale = sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE));
  float Whh_scale = sqrt(2.0 / (HIDDEN_SIZE + HIDDEN_SIZE));
  float Why_scale = sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE));

  for (int i = 0; i < HIDDEN_SIZE; i++) {
    for (int j = 0; j < INPUT_SIZE; j++) {
      Wxh[i][j] = Wxh_scale * (random(1000) / 1000.0 * 2 - 1);
    }
  }

  for (int i = 0; i < HIDDEN_SIZE; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      Whh[i][j] = Whh_scale * (random(1000) / 1000.0 * 2 - 1);
    }
  }

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      Why[i][j] = Why_scale * (random(1000) / 1000.0 * 2 - 1);
    }
  }

  // Initialiser les biais à zéro
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    bh[i] = 0;
  }

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    by[i] = 0;
  }
}
// Passe avant du RNN
void rnn_forward(float *input, float *hidden, float *output) {
  // Mise à jour de l'état caché
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    hidden[i] = 0;
    for (int j = 0; j < INPUT_SIZE; j++) {
      hidden[i] += Wxh[i][j] * input[j];
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      hidden[i] += Whh[i][j] * hidden[j];
    }
    hidden[i] += bh[i];
    hidden[i] = sigmoid(hidden[i]);
  }

  // Calcul de la sortie
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    output[i] = 0;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      output[i] += Why[i][j] * hidden[j];
    }
    output[i] += by[i];
    output[i] = sigmoid(output[i]);
  }
}

// Mise à jour des poids et biais
void rnn_backward(float *input, float *hidden, float *output, float *target) {
  float output_error[OUTPUT_SIZE];
  float hidden_error[HIDDEN_SIZE];

  // Calculer l'erreur de sortie
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    output_error[i] = (output[i] - target[i]) * dsigmoid(output[i]);
  }

  // Mettre à jour les poids Why et les biais by
  for (int i = 0; i < OUTPUT_SIZE; i++) {
      for (int j = 0; j < HIDDEN_SIZE; j++) {
      Why[i][j] -= LR * output_error[i] * hidden[j];
    }
    by[i] -= LR * output_error[i];
  }

  // Calculer l'erreur cachée
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    hidden_error[i] = 0;
    for (int j = 0; j < OUTPUT_SIZE; j++) {
      hidden_error[i] += output_error[j] * Why[j][i];
    }
    hidden_error[i] *= dsigmoid(hidden[i]);
  }

  // Mettre à jour les poids Wxh, Whh et les biais bh
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    for (int j = 0; j < INPUT_SIZE; j++) {
      Wxh[i][j] -= LR * hidden_error[i] * input[j];
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      Whh[i][j] -= LR * hidden_error[i] * hidden[j];
    }
    bh[i] -= LR * hidden_error[i];
  }
}

void setup() {
  // Initialiser la communication série
  Serial.begin(115200);
  // Initialiser les poids et les biais avec des valeurs appropriées
  initialize_weights_biases();
}

void loop() {
  // Préparer les données d'entraînement (entrée et sortie attendue)
  float input[INPUT_SIZE] = {1, 0, 1}; // Exemple d'entrée
  float target[OUTPUT_SIZE] = {1, 0}; // Exemple de sortie attendue

  // Exécuter une passe avant
  float hidden[HIDDEN_SIZE];
  float output[OUTPUT_SIZE];
  rnn_forward(input, hidden, output);

  // Afficher l'erreur avant la mise à jour
  float error_before = 0;
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    error_before += 0.5 * pow(target[i] - output[i], 2);
  }
  Serial.print("Erreur avant la mise à jour : ");
  Serial.println(error_before);

  // Effectuer une mise à jour des poids et des biais (passe arrière)
  rnn_backward(input, hidden, output, target);

  // Exécuter une autre passe avant pour vérifier la mise à jour
  rnn_forward(input, hidden, output);

  // Afficher l'erreur après la mise à jour
  float error_after = 0;
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    error_after += 0.5 * pow(target[i] - output[i], 2);
  }
  Serial.print("Erreur après la mise à jour : ");
  Serial.println(error_after);

  delay(1000);
}
