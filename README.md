# Multi-Layer Perceptron per la Classificazione Multimodale

> Progetto finale per l’esame di **Intelligenza Artificiale**  
> Corso di laurea magistrale LM-43 - Università di Catania  
> Anno accademico 2024/2025

---

## 🧠 Obiettivo del progetto

Il progetto mira a **classificare immagini di azioni umane** sfruttando **tecniche di classificazione multimodale**, ovvero combinando l'informazione visuale (immagine) e quella testuale (descrizione). Il modello è addestrato a riconoscere l'azione svolta (es. "dancing", "eating", "sleeping"...) osservando sia l’immagine che una descrizione coerente.

---

## 📂 Dataset utilizzato

È stato utilizzato un dataset annotato (https://huggingface.co/datasets/Bingsu/Human_Action_Recognition) contenente:

- Immagini raffiguranti 15 classi di azioni umane
- Descrizioni testuali sintetiche per ogni immagine (es. `a photo of a person smiling with an open mouth`)

Ogni immagine è associata a una label e a una frase coerente per abilitare la classificazione multimodale tramite modello CLIP.

---

## 📌 Task affrontato

Il task è un **multi-class classification**, cioè assegnare ad ogni immagine una **sola classe** tra 15 possibili azioni. Si tratta di un **problema supervisionato**, dove le etichette vere sono note.

---

## 🧩 Come viene affrontato

1. **Preprocessing**:
   - Le immagini sono preprocessate in formato compatibile con CLIP
   - Il testo è tokenizzato e normalizzato
2. **Feature Extraction**:
   - Vengono estratti **embedding visivi** e **embedding testuali** usando il modello CLIP
   - I due embedding vengono **concatenati** per ottenere una rappresentazione multimodale di ogni esempio
3. **Classificazione**:
   - La rappresentazione multimodale viene passata a un **MLP** (Multi-Layer Perceptron) per predire la classe
   - Ottimizzazione con **SGD**, loss = cross entropy

---

## 🤖 Modello utilizzato: CLIP + MLP

### 🔍 Cos'è CLIP?

[CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (Contrastive Language-Image Pretraining) è un **modello multimodale pre-addestrato** da OpenAI. Il suo obiettivo è mappare **immagini** e **testi** nello stesso spazio vettoriale in modo che descrizioni testuali corrette siano **vicine** alle immagini corrispondenti.

- Encoder immagine: `ViT-B/32` (Vision Transformer)
- Encoder testo: Transformer classico

CLIP viene usato **come estrattore di feature**, e non viene riaddestrato. Le feature vengono poi usate per la classificazione.

---

## 🧪 Tecniche utilizzate

- **Classificazione multimodale** (immagine + testo)
- **Feature fusion** via concatenazione
- **MLP con ReLU** come classificatore
- **Ottimizzazione** con Stochastic Gradient Descent
- **Early stopping** 
- **Valutazione** con Accuracy e Confusion Matrix

---

## 📚 Librerie principali

- `torch`, `torchvision`
- `transformers` (HuggingFace)
- `sklearn`
- `matplotlib`, `seaborn`

---

## 📈 Risultati ottenuti

- **Accuracy finale su validation set**: ~99.7%
- **Loss** in costante diminuzione già dalle prime epoche
- **Matrice di confusione** perfetta: tutte le 15 classi classificate senza errori nel validation set

---

## 📊 Grafici

### Andamento della loss e accuratezza

<img width="1189" height="490" alt="download" src="https://github.com/user-attachments/assets/3686ba98-d75b-453f-995d-c69205253ae4" />

---

### Matrice di confusione

<img width="791" height="790" alt="download (1)" src="https://github.com/user-attachments/assets/6115c26e-41b1-4630-a0a9-d9629a5bc151" />



