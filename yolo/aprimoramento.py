"""
Conversor Kaggle Kuzushiji → YOLO
Script Completo - Pronto para Usar
Matheus - Projeto Integrador 2026

Uso:
  python script.py --input "seu/caminho/kaggle" --output "seu/caminho/saida"
"""

import pandas as pd
import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KaggleToYOLOConverter:
    """Conversor de dataset Kaggle para formato YOLO"""
    
    def __init__(self, input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Inicializa o conversor
        
        Args:
            input_dir: Diretório com os arquivos do Kaggle (train.csv, unicode_translation.csv, etc)
            output_dir: Diretório de saída para o dataset YOLO
            train_ratio: Proporção de treino (default 0.7)
            val_ratio: Proporção de validação (default 0.15)
            test_ratio: Proporção de teste (default 0.15)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validar proporções
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "As proporções devem somar 1.0"
        
        # Diretórios principais
        self.train_csv = self.input_dir / "train.csv"
        self.train_images_dir = self.input_dir / "train_images"
        self.test_images_dir = self.input_dir / "test_images"
        self.unicode_file = self.input_dir / "unicode_translation.csv"
        
        # Criar estrutura de saída
        self.output_images_dir = self.output_dir / "images"
        self.output_labels_dir = self.output_dir / "labels"
        
        self.train_dir = self.output_images_dir / "train"
        self.val_dir = self.output_images_dir / "val"
        self.test_dir = self.output_images_dir / "test"
        
        self.train_labels = self.output_labels_dir / "train"
        self.val_labels = self.output_labels_dir / "val"
        self.test_labels = self.output_labels_dir / "test"
        
        # Unicode mapping
        self.unicode_map = {}
        
    def create_directories(self):
        """Criar estrutura de diretórios"""
        logger.info("Criando estrutura de diretórios...")
        
        for directory in [self.train_dir, self.val_dir, self.test_dir,
                         self.train_labels, self.val_labels, self.test_labels]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"✓ Diretórios criados em: {self.output_dir}")
    
    def load_unicode_mapping(self):
        """Carregar mapeamento Unicode -> char"""
        logger.info("Carregando mapeamento Unicode...")
        
        try:
            df = pd.read_csv(self.unicode_file)
            self.unicode_map = dict(zip(df['Unicode'], range(len(df))))
            logger.info(f"✓ {len(self.unicode_map)} caracteres únicos carregados")
        except Exception as e:
            logger.warning(f"⚠ Erro ao carregar unicode_translation.csv: {e}")
            logger.info("  Usando mapeamento genérico (todos como classe 0)")
    
    def parse_labels(self, label_string):
        """
        Parse das labels do Kaggle
        Formato: U+XXXX x y width height U+YYYY x y width height ...
        
        Args:
            label_string: String com as anotações
            
        Returns:
            Lista de tuplas (class_id, x, y, width, height)
        """
        boxes = []
        tokens = label_string.split()
        
        i = 0
        while i < len(tokens):
            if tokens[i].startswith('U+'):
                unicode_char = tokens[i]
                
                # Pega os próximos 4 valores (x, y, width, height)
                if i + 4 < len(tokens):
                    try:
                        x = int(tokens[i + 1])
                        y = int(tokens[i + 2])
                        w = int(tokens[i + 3])
                        h = int(tokens[i + 4])
                        
                        # Obter class_id do mapeamento
                        class_id = self.unicode_map.get(unicode_char, 0)
                        
                        boxes.append((class_id, x, y, w, h))
                        i += 5
                    except (ValueError, IndexError):
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        return boxes
    
    def normalize_box(self, x, y, w, h, img_width, img_height):
        """
        Normalizar bounding box para formato YOLO
        YOLO usa: x_center, y_center, width, height (todos normalizados 0-1)
        
        Args:
            x, y, w, h: coordenadas em pixels
            img_width, img_height: dimensões da imagem
            
        Returns:
            Tupla (x_center_norm, y_center_norm, width_norm, height_norm)
        """
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width_norm = w / img_width
        height_norm = h / img_height
        
        # Clipping para garantir valores válidos
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width_norm = np.clip(width_norm, 0, 1)
        height_norm = np.clip(height_norm, 0, 1)
        
        return x_center, y_center, width_norm, height_norm
    
    def process_image(self, image_id, label_string):
        """
        Processar uma imagem e suas anotações
        
        Args:
            image_id: ID da imagem
            label_string: String com as anotações
            
        Returns:
            Dicionário com dados processados ou None se falhar
        """
        # Procurar a imagem
        image_path = None
        
        # Tentar encontrar em train_images
        for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            candidate = self.train_images_dir / f"{image_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if not image_path:
            # Tentar em test_images
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                candidate = self.test_images_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        
        if not image_path:
            logger.warning(f"⚠ Imagem não encontrada: {image_id}")
            return None
        
        try:
            # Abrir imagem
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Parse das bounding boxes
            boxes = self.parse_labels(label_string)
            
            if not boxes:
                logger.warning(f"⚠ Nenhuma anotação encontrada para: {image_id}")
                return None
            
            # Normalizar boxes para formato YOLO
            yolo_annotations = []
            for class_id, x, y, w, h in boxes:
                x_norm, y_norm, w_norm, h_norm = self.normalize_box(x, y, w, h, img_width, img_height)
                yolo_annotations.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            return {
                'image_path': image_path,
                'image_id': image_id,
                'img': img,
                'annotations': yolo_annotations,
                'num_boxes': len(boxes)
            }
        
        except Exception as e:
            logger.error(f"✗ Erro ao processar {image_id}: {e}")
            return None
    
    def save_to_yolo_structure(self, data, split):
        """
        Salvar imagem e anotações na estrutura YOLO
        
        Args:
            data: Dicionário com dados processados
            split: 'train', 'val' ou 'test'
        """
        if split == 'train':
            img_dir = self.train_dir
            label_dir = self.train_labels
        elif split == 'val':
            img_dir = self.val_dir
            label_dir = self.val_labels
        else:
            img_dir = self.test_dir
            label_dir = self.test_labels
        
        # Salvar imagem
        image_id = data['image_id']
        output_image_path = img_dir / f"{image_id}.jpg"
        data['img'].convert('RGB').save(output_image_path, 'JPEG', quality=95)
        
        # Salvar anotações
        output_label_path = label_dir / f"{image_id}.txt"
        with open(output_label_path, 'w', encoding='utf-8') as f:
            for annotation in data['annotations']:
                f.write(annotation + '\n')
    
    def convert(self):
        """Executar a conversão completa"""
        logger.info("="*60)
        logger.info("Iniciando conversão Kaggle -> YOLO")
        logger.info("="*60)
        
        # Criar diretórios
        self.create_directories()
        
        # Carregar mapeamento Unicode
        self.load_unicode_mapping()
        
        # Carregar train.csv
        logger.info("Carregando train.csv...")
        try:
            df = pd.read_csv(self.train_csv)
            logger.info(f"✓ {len(df)} imagens encontradas no train.csv")
        except Exception as e:
            logger.error(f"✗ Erro ao carregar train.csv: {e}")
            return False
        
        # Split treino/validação/teste
        logger.info(f"Fazendo split: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        
        train_data, temp_data = train_test_split(
            df, test_size=(self.val_ratio + self.test_ratio),
            random_state=42
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
            random_state=42
        )
        
        logger.info(f"  Train: {len(train_data)} imagens")
        logger.info(f"  Val:   {len(val_data)} imagens")
        logger.info(f"  Test:  {len(test_data)} imagens")
        
        # Processar e salvar
        splits = [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]
        
        total_processed = 0
        total_skipped = 0
        
        for split_name, split_df in splits:
            logger.info(f"\nProcessando {split_name.upper()}...")
            
            split_processed = 0
            split_skipped = 0
            
            for idx, row in tqdm(split_df.iterrows(), total=len(split_df), 
                                desc=f"  {split_name}"):
                image_id = row['image_id']
                labels = row['labels']
                
                # Processar imagem
                data = self.process_image(image_id, labels)
                
                if data:
                    self.save_to_yolo_structure(data, split_name)
                    split_processed += 1
                    total_processed += 1
                else:
                    split_skipped += 1
                    total_skipped += 1
            
            logger.info(f"  ✓ {split_name}: {split_processed} processadas, {split_skipped} puladas")
        
        logger.info(f"\nResumo total:")
        logger.info(f"  ✓ Processadas: {total_processed}")
        logger.info(f"  ✗ Puladas:     {total_skipped}")
        
        # Criar data.yaml
        self.create_data_yaml(total_processed)
        
        logger.info("="*60)
        logger.info("✓ Conversão concluída com sucesso!")
        logger.info(f"  Dataset YOLO salvo em: {self.output_dir}")
        logger.info("="*60)
        
        return True
    
    def create_data_yaml(self, total_images):
        """Criar arquivo data.yaml para YOLO"""
        logger.info("Criando data.yaml...")
        
        # Se temos mapeamento Unicode, criar lista de nomes
        if self.unicode_map:
            # Inverter o mapa para obter unicode_char -> class_id
            reverse_map = {v: k for k, v in self.unicode_map.items()}
            
            # Criar lista ordenada de nomes
            names = {}
            for class_id in sorted(reverse_map.keys()):
                unicode_char = reverse_map[class_id]
                names[class_id] = unicode_char
        else:
            # Mapeamento genérico
            names = {0: 'character'}
        
        # Contar imagens por split
        train_count = len(list(self.train_dir.glob('*.jpg'))) + len(list(self.train_dir.glob('*.png')))
        val_count = len(list(self.val_dir.glob('*.jpg'))) + len(list(self.val_dir.glob('*.png')))
        test_count = len(list(self.test_dir.glob('*.jpg'))) + len(list(self.test_dir.glob('*.png')))
        
        # Criar conteúdo do YAML
        yaml_content = f"""# Kuzushiji Recognition Dataset - YOLO Format
# Convertido do Kaggle dataset
# Data: 2026

path: {self.output_dir.absolute()}

train: images/train
val: images/val
test: images/test

# Número de classes
nc: {len(names)}

# Nomes das classes
names:
"""
        
        for class_id in sorted(names.keys()):
            yaml_content += f"  {class_id}: '{names[class_id]}'\n"
        
        yaml_content += f"""
# Estatísticas
total_images: {train_count + val_count + test_count}
train_images: {train_count}
val_images: {val_count}
test_images: {test_count}
"""
        
        # Salvar
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"✓ data.yaml criado: {yaml_path}")
    
    def get_dataset_info(self):
        """Retornar informações sobre o dataset"""
        return {
            'output_dir': self.output_dir,
            'num_classes': len(self.unicode_map) if self.unicode_map else 1,
            'train_dir': self.train_dir,
            'val_dir': self.val_dir,
            'test_dir': self.test_dir,
        }


def main():
    # Caminhos fixos
    input_dir = r'C:\Users\kauan\Downloads\kuzushiji-recognition (1)'
    output_dir = r'C:\Faculdade\Projeto integrador'

    # Criar conversor
    converter = KaggleToYOLOConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Executar conversão
    success = converter.convert()

    if success:
        info = converter.get_dataset_info()

        print("\n" + "=" * 60)
        print("✓ Dataset YOLO criado com sucesso!")
        print(f"Localização: {info['output_dir']}")
        print(f"Classes: {info['num_classes']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
