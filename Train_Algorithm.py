import numpy as np
from Encoder import Encoder
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



class TextDataset(Dataset):
    def __init__(self, tokenized_text, max_seq_len):
        self.data = []
        for i in range(0, len(tokenized_text), max_seq_len):
            self.data.append(tokenized_text[i:i+max_seq_len])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # Pad auf `max_seq_len` auffüllen, wenn die Sequenz kürzer ist
        if len(sequence) < max_seq_len:
            sequence = sequence + [tokenizer.pad_token_id] * (max_seq_len - len(sequence))
        
        
        return torch.tensor(sequence)
    

def apply_mlm_masking(input_ids, device, mask_prob):
    maskable_tokens = (input_ids != tokenizer.pad_token_id) * (input_ids != 101) * (input_ids != 102)
    
    random_values = torch.rand(input_ids.shape).to(device)
    
    mask = (random_values < mask_prob) & maskable_tokens
    
    masked_batch = input_ids.clone()
    masked_batch[mask] = 103
    #print(masked_batch)
    
    # labelerstellung:
    labels = torch.where(masked_batch == 103, input_ids, torch.full_like(input_ids, -100))
    labels = labels.long()
    #print("Labels: ", labels)
    return masked_batch, labels


def save_model():
    
    model_saving_path = "Your path here" 
    
    torch.save({
    'model_state_dict': model.state_dict(),
    #'optimizer_state_dict': optimizer.state_dict(),            # Auskommentiert, da das Speichern des Optimizers nicht notwendig ist.
    }, model_saving_path)
    
    print("Das Modell wurde erfolgreich gespeichert")
    

def load_model():
    
    model_loading_path = "Your path here"
    # Checkpoint laden
    checkpoint = torch.load(model_loading_path, weights_only = True)
    
    # Modellzustand laden
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizer-Zustand laden
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Das Modell wurde erfolgreich aus '{model_loading_path}' geladen.")
    return model
    
    

def get_progressive_datasets(dataset_amount, tokenizer, start_index):

    paths = []
    tokenized_texts = []
    #print("Das ist der Startindex: ", start_index)                # Überprüfung, ob der Index korrekt startet
    for i in range(start_index, (start_index + dataset_amount)):
        paths.append(f"Your path here, Data_{i+1}.txt")
    
    #for path in paths:
        #print("Ausgewählte Datei: ", path)
        
    for path in paths:
        
        with open(fr"{path}", "r", encoding = "utf-8")as file:
            tokenized_texts.append(tokenizer.encode(file.read(), truncation=False, add_special_tokens=False))            
        
    
            
    return tokenized_texts
    
    
# Bert-base-german-cased als Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

# Dies kann sehr gerne angepasst werden. Meine Grafikkarte (RTX 4060) hat mit 8GB VRAM begrenzten Speicher.
batch_size = 14
max_seq_len = 256
dataset_amount = 5
vocab_size = tokenizer.vocab_size
embedding_dim = 516
num_heads = 12
mask_prob = 0.15
lr = 1e-4

start_index = 0
counter = 0

epochs = 2000

targeted_accuracy = 81
final_accuracy = 91
cycles = 65
stabilization_cycles = 5      



model = Encoder(vocab_size= vocab_size, embedding_dim = embedding_dim, max_seq_len = max_seq_len, num_heads = num_heads)


criterion = nn.CrossEntropyLoss(ignore_index=-100)        # Loss
optimizer_initialize = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay = 0.01)    # Optimizer

model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Modell auf GPU verschieben
model = model.to(device)
model.train()

for state in optimizer_initialize.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)



datasets = []
dataloaders_progressive = []
dataloaders_stabilization = []

for cycle in range(cycles):

    
    print("Actual Cycle: ", cycle + 1)
    
    datasets.clear()
    
    
    tokenized_texts = get_progressive_datasets(dataset_amount, tokenizer, start_index)
        
    for tokenized_text in tokenized_texts:
        datasets.append(TextDataset(tokenized_text, max_seq_len = max_seq_len))
     
    for dataset in datasets:
        usable_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle = True)
        
        dataloaders_progressive.append(usable_dataloader)
        dataloaders_stabilization.append(usable_dataloader)
    
         
    # Progressive Cycle
    print("Progressive Cycle started")
    for current_max in range(2, dataset_amount + 1):
              
        chosen_dataloaders = dataloaders_progressive[:current_max]
        dataloader_counter = 0
        
        for dataloader in chosen_dataloaders:
            
            optimizer = optimizer_initialize
            epoch_counter = 0
            print("Aktuelle Datei des Progressive Cycles: ", dataloader_counter+1)
            
            for epoch in range(epochs):
                
                total_correct = 0
                total_tokens = 0
                for batch in dataloader:
                    
                    batch = batch.to(device)  # Batch auf GPU verschieben
                    masked_batch, labels = apply_mlm_masking(batch, device, mask_prob)
                    
                    optimizer.zero_grad()    # Gradienten zurücksetzen

                    outputs, _ = model(masked_batch)
                    logits = outputs.view(-1, vocab_size)  # Logits für den Loss vorbereiten
                        
                    labels = labels.view(-1)
                    
                    loss = criterion(logits, labels)  # Verlust berechnen
                    loss.backward()          # Gradientenberechnung
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                    optimizer.step()         # Parameterupdate
        
                    mask = labels != -100
                    predicted = torch.argmax(logits, dim=-1)  # Vorhersagen (Index des höchsten Logits)
                    correct = (predicted == labels) & mask  # Zählen der richtigen Vorhersagen
                    correct = correct.sum().item()
                    total_correct += correct
                    total_tokens += mask.sum()  # Gesamtzahl der Tokens im Batch
                
                epoch_counter += 1
                accuracy = total_correct / total_tokens * 100  # Accuracy in Prozent
                #print(f"accuracy: {accuracy}, Loss: {loss.item()}")
                #if epoch_counter == 1:
                    #print(f"Erste Epoche für Datei {dataloader_counter+1} abgeschlossen: Loss: {loss.item()} - Accuracy: {accuracy:.2f}%" )
                
                if accuracy >= targeted_accuracy:
                    print("Anzahl derbenötigten Epochen: ", epoch_counter)
                    #print(f"Finale Epoche abgeschlossen Loss: {loss.item()} - Accuracy: {accuracy:.2f}%")
                    break       
           
            dataloader_counter += 1
             
    # Freigabe der Progressive-List, da nun der Stabilization-Cycle Startet
    dataloaders_progressive.clear()       
    torch.cuda.empty_cache()
    
    # Stabilization Cycle       
    # Stabilization_cycle Part 1
    
    print("Stabilization Cycle started")
        
    for stabilization_cycle in range(stabilization_cycles): 
        smaller_lr = lr * 0.9      # Hier wird die Lernrate reduziert, damit 'Gewichtsspitzen' wegen der unterschiedlichen Datensätze geglättet werden.
        print(smaller_lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=smaller_lr, betas=(0.9, 0.98), weight_decay = 0.01)
        
        dataloader_counter = 0
        global_epoch_counter = 0    
           
        for dataloader in dataloaders_stabilization:
            
            epoch_counter = 0
            print("Aktuelle Datei des Stabilization Cycles: ", dataloader_counter+1)
            for epoch in range(epochs):
                
                total_correct = 0
                total_tokens = 0
                for batch in dataloader:
                    
                    batch = batch.to(device)  # Batch auf GPU verschieben
                    masked_batch, labels = apply_mlm_masking(batch, device, mask_prob)
                    
                    optimizer.zero_grad()    # Gradienten zurücksetzen

                    outputs, _ = model(masked_batch)
                    logits = outputs.view(-1, vocab_size)  # Logits für den Loss vorbereiten
                        
                    labels = labels.view(-1)
                    
                    loss = criterion(logits, labels)  # Verlust berechnen
                    loss.backward()          # Gradientenberechnung
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                    optimizer.step()         # Parameterupdate
        
                    mask = labels != -100
                    predicted = torch.argmax(logits, dim=-1)  # Vorhersagen (Index des höchsten Logits)
                    correct = (predicted == labels) & mask  # Zählen der richtigen Vorhersagen
                    correct = correct.sum().item()
                    total_correct += correct
                    total_tokens += mask.sum()  # Gesamtzahl der Tokens im Batch
                   
                    
                accuracy = total_correct / total_tokens * 100  # Accuracy in Prozent
                epoch_counter += 1
                
                
                
                if accuracy >= final_accuracy:
                    global_epoch_counter += epoch_counter
                    print("Anzahl derbenötigter Epochen: ", epoch_counter)
                    break
                
            dataloader_counter += 1    
            torch.cuda.empty_cache()
            
        
            
        with open(r"C:\Users\Alex\Desktop\Python\KIs\Natural_Language_Processing\Encoder\LOGs.txt", "a")as file:
            file.write(f"Cycle: {cycle + 1}      Iteration: {stabilization_cycle + 1}\n")
            file.write(f"Durchschnittliche Epochen der Stabilisation: {global_epoch_counter / len(dataloaders_stabilization)}\n")
                
            
        
        
    save_model()  
              
    del optimizer, accuracy, epoch_counter, total_correct, total_tokens, loss, dataloader_counter          
    torch.cuda.empty_cache()
    start_index += dataset_amount

print("CycleTrainer beendet.")


        
