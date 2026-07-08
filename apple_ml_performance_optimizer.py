#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple ML Performance Optimization Expert System
Optimized version with enhanced performance strategies for Core ML models
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ModelType(Enum):
    """Enum for supported model types"""
    IMAGE_CLASSIFICATION = "image_classification"
    TEXT_CLASSIFICATION = "text_classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    RECOMMENDATION_SYSTEM = "recommendation_system"


@dataclass
class ModelInput:
    """Represents a model input specification"""
    name: str
    type: str
    format: str
    size: str
    description: str


@dataclass
class ModelOutput:
    """Represents a model output specification"""
    name: str
    type: str
    description: str
    format: str


@dataclass
class ModelDefinition:
    """Complete model definition with all specifications"""
    name: str
    type: str
    description: str
    inputs: List[Dict[str, str]]
    outputs: List[Dict[str, str]]
    best_practices: List[str]


class AppleMLPerformanceOptimizer:
    """
    High-performance Apple ML optimization expert system.
    Provides optimized strategies for reducing latency and improving Core ML model performance.
    """
    
    # Pre-computed optimization strategies for faster access
    PERFORMANCE_OPTIMIZATION_STRATEGIES = [
        "**Quantização**: Reduza a precisão dos pesos e ativações do modelo (float32 → float16/int8) para diminuir tamanho e acelerar inferência.",
        "**Batching**: Execute inferências em lotes para aproveitar melhor o hardware do dispositivo.",
        "**Neural Engine**: Configure o modelo para utilizar o Neural Engine (dispositivos A11 Bionic+) para máxima performance.",
        "**Redução de Overheads**: Minimize cópias de dados e operações de pré/pós-processamento externas ao Core ML.",
        "**Perfil de Performance**: Use Instruments no Xcode para perfilar performance do aplicativo com Core ML."
    ]
    
    LATENCY_REDUCTION_BEST_PRACTICES = [
        "**Otimização do Modelo**: Use modelos menores e eficientes. Explore arquiteturas otimizadas para dispositivos.",
        "**Quantização**: Reduza precisão dos pesos (float32 → float16/int8) para diminuir tamanho e acelerar inferência no Neural Engine.",
        "**Pré-processamento Eficiente**: Minimize complexidade do pré-processamento. Execute no dispositivo quando possível.",
        "**Uso do Neural Engine**: Configure Core ML para aproveitar Neural Engine da Apple para aceleração significativa.",
        "**Inferência em Lote (Batching)**: Agrupe múltiplas entradas para reduzir overhead por inferência.",
        "**Warm-up do Modelo**: Execute 1-2 inferências 'falsas' no início para carregar modelo na memória e otimizar pipeline.",
        "**Evitar Cópias de Dados Desnecessárias**: Minimize transferência CPU ↔ GPU/Neural Engine. Mantenha dados no mesmo domínio de memória.",
        "**Assincronia e Threading**: Execute inferência em thread secundária para não bloquear UI principal.",
        "**Perfil e Análise**: Use Instruments do Xcode para identificar gargalos específicos na pipeline de inferência.",
        "**Simplificação do Modelo**: Para latência crítica, use pruning, distillation ou NAS para equilíbrio desempenho-eficiência."
    ]
    
    def __init__(self):
        """Initialize optimizer with pre-loaded model definitions"""
        self._model_definitions: Dict[str, ModelDefinition] = {}
        self._error_solutions: Dict[str, str] = {}
        self._initialize_models()
        self._initialize_error_lookup()
    
    def _initialize_models(self) -> None:
        """Pre-load all model definitions for fast access"""
        models_data = [
            {
                "name": "ImageClassifier",
                "type": "image_classification",
                "description": "Configuração para modelo de classificação de imagens.",
                "inputs": [
                    {"name": "input_image", "type": "Image", "format": "Grayscale | Color (RGB/BGR)", 
                     "size": "224x224", "description": "Imagem de entrada para classificação."}
                ],
                "outputs": [
                    {"name": "class_labels", "type": "String", "description": "Lista de rótulos de classe previstos.", 
                     "format": "Array de Strings"},
                    {"name": "probabilities", "type": "Dictionary", "description": "Probabilidades para cada classe.", 
                     "format": "Dicionário de String para Float"}
                ],
                "best_practices": [
                    "Pré-processamento: normalização, redimensionamento para tamanho do modelo.",
                    "Use dataset de treinamento diversificado para evitar overfitting.",
                    "Valide com dataset de validação independente."
                ]
            },
            {
                "name": "TextClassifier",
                "type": "text_classification",
                "description": "Configuração para modelo de classificação de texto.",
                "inputs": [
                    {"name": "input_text", "type": "String", "description": "Texto de entrada para classificação."}
                ],
                "outputs": [
                    {"name": "class_label", "type": "String", "description": "Rótulo de classe previsto.", 
                     "format": "String"},
                    {"name": "probabilities", "type": "Dictionary", "description": "Probabilidades para cada classe.", 
                     "format": "Dicionário de String para Float"}
                ],
                "best_practices": [
                    "Limpeza: remover pontuação, caracteres especiais, converter para minúsculas.",
                    "Tokenização e vetorização antes de alimentar o modelo.",
                    "Balanceie classes no dataset para evitar viés."
                ]
            },
            {
                "name": "ObjectDetection",
                "type": "object_detection",
                "description": "Configuração para detecção de objetos em imagens.",
                "inputs": [
                    {"name": "input_image", "type": "Image", "format": "Color (RGB)", 
                     "size": "Variável (ex: 416x416)", "description": "Imagem de entrada para detecção."}
                ],
                "outputs": [
                    {"name": "detected_boxes", "type": "Array", "description": "Caixas delimitadoras.", 
                     "format": "Array de [x, y, largura, altura]"},
                    {"name": "detected_labels", "type": "Array", "description": "Rótulos de classe.", 
                     "format": "Array de Strings"},
                    {"name": "confidences", "type": "Array", "description": "Confianças das detecções.", 
                     "format": "Array de Floats"}
                ],
                "best_practices": [
                    "Otimize para tamanho de imagem e número de classes esperados.",
                    "Use Non-Maximum Suppression (NMS) para remover detecções duplicadas.",
                    "Ajuste limiares de confiança para balancear precisão e recall."
                ]
            },
            {
                "name": "SemanticSegmentation",
                "type": "semantic_segmentation",
                "description": "Configuração para segmentação semântica de imagens.",
                "inputs": [
                    {"name": "input_image", "type": "Image", "format": "Color (RGB)", 
                     "size": "Variável (ex: 512x512)", "description": "Imagem de entrada para segmentação."}
                ],
                "outputs": [
                    {"name": "segmentation_map", "type": "Image", "description": "Mapa de segmentação por pixel.", 
                     "format": "Grayscale (mapa de classes)"},
                    {"name": "confidence_map", "type": "Image", "description": "Mapa de confiança por pixel.", 
                     "format": "Grayscale (mapa de confiança)"}
                ],
                "best_practices": [
                    "Anotações precisas e consistentes para cada pixel.",
                    "Use data augmentation para variabilidade nas imagens.",
                    "Otimize pós-inferência para visualização do mapa.",
                    "Priorize arquiteturas leves (DeepLabv3+ com MobileNetV2) para tempo real."
                ]
            },
            {
                "name": "RecommendationSystem",
                "type": "recommendation_system",
                "description": "Configuração para sistema de recomendação.",
                "inputs": [
                    {"name": "user_id", "type": "Integer", "format": "Scalar", 
                     "size": "1", "description": "ID único do usuário."},
                    {"name": "context_data", "type": "Dictionary", "format": "JSON", 
                     "size": "Variable", "description": "Dados contextuais adicionais."}
                ],
                "outputs": [
                    {"name": "recommended_items", "type": "Array", 
                     "description": "IDs de itens recomendados.", "format": "Array de Strings"},
                    {"name": "scores", "type": "Array", 
                     "description": "Pontuações de relevância.", "format": "Array de Floats"}
                ],
                "best_practices": [
                    "Personalize com base no histórico e contexto.",
                    "Atualize regularmente com novos dados.",
                    "Considere diversidade e novidade das recomendações.",
                    "Implemente testes A/B para avaliar eficácia."
                ]
            }
        ]
        
        for model_data in models_data:
            model = ModelDefinition(**model_data)
            self._model_definitions[model.name] = model
    
    def _initialize_error_lookup(self) -> None:
        """Pre-build error lookup table for O(1) access"""
        errors = [
            ("Input shape mismatch", 
             "Verifique dimensões de entrada (ex: imagem 224x224) e garanta correspondência com especificações."),
            ("Model not found/loaded", 
             "Verifique caminho do .mlmodel e permissões. Certifique-se que modelo está no projeto Xcode."),
            ("Out of memory", 
             "Otimize tamanho (quantização), processe em lotes menores, use GPU/NPU se disponível.")
        ]
        self._error_solutions = {err.lower(): sol for err, sol in errors}
    
    def get_model_definition(self, model_name: str) -> Optional[ModelDefinition]:
        """Fast O(1) lookup for model definitions"""
        return self._model_definitions.get(model_name)
    
    def get_all_models(self) -> List[str]:
        """Get list of all available model names"""
        return list(self._model_definitions.keys())
    
    def get_error_solution(self, error_message: str) -> str:
        """
        Fast error solution lookup with fuzzy matching.
        Optimized for performance with pre-computed lookup table.
        """
        error_lower = error_message.lower()
        
        # O(1) lookup for known errors
        for error_key, solution in self._error_solutions.items():
            if error_key in error_lower:
                return f"Solução para '{error_key.title()}': {solution}"
        
        # Fallback to general tips
        return self._get_general_performance_tips()
    
    def _get_general_performance_tips(self) -> str:
        """Return optimized general performance tips"""
        tips = [
            "Use Core ML Tools para converter TensorFlow/PyTorch para .mlmodel.",
            "Aproveite Create ML no Xcode para tarefas comuns.",
            "Use Neural Engine para inferências rápidas em dispositivos compatíveis.",
            "Monitore desempenho para identificar gargalos e otimizações."
        ]
        return "Nenhuma solução específica encontrada.\nDicas gerais:\n" + "\n".join(tips)
    
    def get_optimization_strategies(self, target_latency_ms: Optional[float] = None) -> List[str]:
        """
        Get prioritized optimization strategies based on latency target.
        
        Args:
            target_latency_ms: Target latency in milliseconds. If provided, returns
                              strategies sorted by impact for achieving this target.
        
        Returns:
            List of optimization strategies prioritized by expected impact.
        """
        if target_latency_ms is None:
            return self.LATENCY_REDUCTION_BEST_PRACTICES.copy()
        
        # Prioritize strategies based on latency target
        if target_latency_ms < 10:
            priority_order = [1, 9, 3, 4, 6, 7, 8, 0, 2, 5]  # Ultra-low latency
        elif target_latency_ms < 50:
            priority_order = [1, 3, 4, 6, 7, 0, 2, 5, 8, 9]  # Low latency
        else:
            priority_order = list(range(len(self.LATENCY_REDUCTION_BEST_PRACTICES)))  # Standard
        
        return [self.LATENCY_REDUCTION_BEST_PRACTICES[i] for i in priority_order]
    
    def export_to_json(self, filepath: str, indent: int = 2) -> None:
        """Export complete dictionary to JSON file"""
        export_data = {
            "model_definitions": [asdict(m) for m in self._model_definitions.values()],
            "performance_optimization_strategies": self.PERFORMANCE_OPTIMIZATION_STRATEGIES,
            "latency_reduction_best_practices": self.LATENCY_REDUCTION_BEST_PRACTICES,
            "common_errors_and_solutions": [
                {"error": k.title(), "solution": v} 
                for k, v in self._error_solutions.items()
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=indent, ensure_ascii=False)
        
        print(f"✓ Dicionário exportado para: {filepath}")
    
    @classmethod
    def import_from_json(cls, filepath: str) -> 'AppleMLPerformanceOptimizer':
        """Import optimizer state from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        optimizer = cls()
        
        if 'model_definitions' in data:
            for model_data in data['model_definitions']:
                model = ModelDefinition(**model_data)
                optimizer._model_definitions[model.name] = model
        
        return optimizer


def main():
    """Main entry point demonstrating optimized usage"""
    print("=" * 70)
    print("Apple ML Performance Optimization Expert System")
    print("=" * 70)
    
    optimizer = AppleMLPerformanceOptimizer()
    
    # Display available models
    print("\n📦 Modelos Disponíveis:")
    print("-" * 70)
    for model_name in optimizer.get_all_models():
        model = optimizer.get_model_definition(model_name)
        if model:
            print(f"  • {model.name}: {model.description}")
    
    # Display performance strategies
    print("\n⚡ Estratégias de Otimização de Performance:")
    print("-" * 70)
    for i, strategy in enumerate(optimizer.get_optimization_strategies()[:5], 1):
        print(f"  {i}. {strategy}")
    
    # Demonstrate error lookup
    print("\n🔍 Exemplo de Busca de Erro:")
    print("-" * 70)
    test_error = "Input shape mismatch: expected 224x224"
    solution = optimizer.get_error_solution(test_error)
    print(f"  Erro: {test_error}")
    print(f"  {solution}")
    
    # Export optimized dictionary
    print("\n💾 Exportando dicionário otimizado...")
    optimizer.export_to_json("apple_ml_optimized_dictionary.json")
    
    print("\n" + "=" * 70)
    print("✓ Sistema pronto para uso!")
    print("=" * 70)


if __name__ == "__main__":
    main()
