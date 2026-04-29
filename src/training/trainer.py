import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm # Para barras de progreso

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Clase encargada del entrenamiento del modelo de transcripción de piano.

        Args:
            model: Instancia del modelo PianoTranscriptionModel.
            train_dataset: Dataset de entrenamiento.
            val_dataset: Dataset de validación.
            config: Diccionario con hiperparámetros (batch_size, lr, epochs, etc.).
        """
        self.model = model
        self.config = config

        # 1. Configuracion del Dispositivo para su adaptacion a Colab
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"✅ Usando dispositivo: {self.device}")

        # 2. Optimizador (AdamW es el estándar para Transformers)
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )

        # 3. Función de Pérdida (CrossEntropy para clasificación de tokens)
        # Ignoramos el token de padding (0) para que no influya en la pérdida
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) 
        
        # 4. Data Loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)

    def _create_dataloader(self, dataset, shuffle):
        """
        Función auxiliar para crear DataLoaders con padding

        La funcion collate_fn del datatset no es ideal para PyTorch Dataloader
        debido a la longitud variable de la secuencia.

        Usaremos la funcion lambda para el manejo del padding (token 0) de las secuencias de tokens
        
        """
        def pad_sequence(batch):
            # EN el flujo de datos del Dataset, cada muestra es un diccionario con 'audio' y 'tokens'
            audios = [item['audio'] for item in batch]
            tokens = [item['tokens'] for item in batch]

            # 1. Padding de Tokens (MIDI)
            tokens_padded = nn.utils.rnn.pad_sequence(
                tokens,
                batch_first=True,
                padding_value=0  # Token de padding
            )

            # 2. Padding de Audio (Espectrogramas Mel)
            # El espectrograma es 3D: [1, n_mels, time].Lo padeamos en la dimension de tiempo (la ultima)
            max_time = max(a.shape[-1] for a in audios)
            audios_padded = torch.zeros(len(audios), audios[0].shape[0], audios[0].shape[1], max_time)
            for i, a in enumerate(audios):
                audios_padded[i, :, :, :a.shape[-1]] = a

            return {
                'audio': audios_padded,
                'tokens': tokens_padded
            }
        
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            collate_fn=pad_sequence, # Usamos nuestra función de padding
            num_workers=self.config['num_workers']
        )
    

    def train_epoch(self):
        """
        Ejecuta una época completa de entrenamiento.
        """
        self.model.train()
        total_loss = 0

        # Usamos tqdm para visualizar el progreso en Colab
        progress_bar = tqdm(self.train_loader, desc="Entrenando")

        for batch in progress_bar:
            audio = batch['audio'].to(self.device)  # [Batch, 1, n_mels, Time]

            # TGT_IN: Tokens de entrada (de 0 a N-1)
            tgt_in = batch['tokens'][:, :-1].to(self.device)

            # TGT_OUT: Tokens objetivo (de 1 a N)
            tgt_out = batch['tokens'][:, 1:].to(self.device)

            # 1. Poner a 0 los gradientes
            self.optimizer.zero_grad()

            # 2. Forward Pass
            logits = self.model(audio, tgt_in)  # [Batch, Seq_Len-1, Vocab_Size]

            # 3. Calcular Pérdida
            # PyTorch CrossEntropy espera [N, C, D1...], así que transponemos los logits y el target
            loss = self.criterion(
                logits.transpose(1, 2), # Lo reordenamos a [Batch, Vocab_Size, Seq_Len-1]
                tgt_out # [Batch, Seq_Len-1]
            )

            # 4. Backward Pass
            loss.backward()

            # 5. Actualizar Pesos
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)
    

    @torch.no_grad()
    def validate(self):
        """
        Ejecuta un ciclo de validación

        """
        self.model.eval()
        total_loss = 0

        progress_bar = tqdm(self.val_loader, desc="Validando")

        for batch in progress_bar:
            audio = batch['audio'].to(self.device)  # [Batch, 1, n_mels, Time]
            tgt_in = batch['tokens'][:, :-1].to(self.device)
            tgt_out = batch['tokens'][:, 1:].to(self.device)

            # Forward Pass
            logits = self.model(audio, tgt_in)  # [Batch, Seq_Len-1, Vocab_Size]

            # Calcular Pérdida
            loss = self.criterion(
                logits.transpose(1, 2), # [Batch, Vocab_Size, Seq_Len-1]
                tgt_out # [Batch, Seq_Len-1]
            )
            total_loss += loss.item()
            progress_bar.set_postfix({'Val Loss': f'{loss.item():.4f}'})

        return total_loss / len(self.val_loader)
    

    def train(self, num_epochs, checkpoint_path='model_checkpoint.pt'):
        """
        Bucle principal de entrenamiento

        """
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\n=== Época {epoch+1}/{num_epochs} ===")

            # 1. Entrenamiento
            train_loss = self.train_epoch()
            
            # 2. Validación
            val_loss = self.validate()

            print(f"📊 Pérdida de Entrenamiento: {train_loss:.4f} | Pérdida de Validación: {val_loss:.4f}")

            # Guardar el mejor modelo basado en la pérdida de validación
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"⭐ Pérdida de validación mejorada. Guardando modelo en {checkpoint_path}")
                torch.save(self.model.state_dict(), checkpoint_path)
                