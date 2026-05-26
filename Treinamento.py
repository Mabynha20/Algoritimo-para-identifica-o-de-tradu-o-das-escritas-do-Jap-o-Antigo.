from ultralytics import YOLO
import multiprocessing as mp

def treinar_modelo():
    """Função para fazer fine-tuning do modelo"""
    
    # Carregue seu melhor modelo treinado
    model = YOLO(r"C:\Faculdade\Conteudos\Projeto integrador\Teste_YOLO\Fine_Tuning\weights\last.pt")

    results = model.train(
        data=r"C:\Faculdade\Conteudos\Projeto integrador\Teste_YOLO\Aprimoramento\data.yaml",
        epochs=200,         # ← REDUZIR (já treinou 500 antes!)
        imgsz=416,
        batch=8,
        patience=15,        # ← Reduzir também (converge mais rápido)
        lr0=0.0005,         # ← MUITO IMPORTANTE! Reduzir de 0.001 para 0.0001
        project=r"C:\Faculdade\Conteudos\Projeto integrador\Teste_YOLO",
        name="Fine_Tuning",  # ← Nome diferente para diferenciar
        exist_ok=True,
        device=0,
        workers=0,
        amp = False,
        cache = True, 
        cos_lr= True,
        resume=True
    ) 
    
    print("\n✓ Fine-tuning concluído!")
    print(f"Melhor modelo: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    mp.freeze_support()
    treinar_modelo()