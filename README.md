# Proyecto-RL: From Solo to Squad: Extending NGU's to Multi-Agent RL

Este repositorio contiene las implementaciones, experimentos y análisis utilizados para el paper sobre la extensión del algoritmo **Never Give Up (NGU)** a entornos multi-agente.

---

## Estructura del repositorio

- **BasicDQN/**  
  Implementación base de DQN sobre la cual se construyeron los experimentos posteriores.

- **Analisis/**  
  Scripts y notebooks de validación que comprueban que la implementación de DQN funciona correctamente.

- **Experiments/**  
  Investigación preliminar con **MultiDQN** en diferentes ambientes antes de extender NGU.

- **NGU/**  
  Código completo necesario para correr **Never Give Up (NGU)** en entornos de un solo agente.

- **NGUMultiAgent/**  
  Extensión de NGU para múltiples agentes.  
  - `NGUMulti.py`: implementación principal de Multi-NGU.  
  - `save_runs_simple_tag_multi_dqn_shared_buffer.ipynb`: experimentos de baseline con Multi-DQN usando buffer compartido.  
  - `save_runs_simple_tag_multi_dqn.ipynb`: experimentos de baseline con Multi-DQN sin buffer compartido.  
  - `save_runs_simple_tag_shared_buffer.ipynb`: experimentos de Multi-NGU con buffer compartido (se puede desactivar en la celda respectiva).  
  - `save_runs_simple_tag_shared_buffer_heterogenous_beta.ipynb`: experimentos con la variante de **β heterogéneo**.  

- **NGUMultiNovelty/**  
  Extensión de NGU con **compartición de novedad**.  
  - `NGUMulti.py`: implementación principal de Multi-NGU con novedad compartida.  
  - `contrastive.py`: módulo para entrenar el embedding de estados usando aprendizaje contrastivo.  
  - `cosine_registry.py`: utilidades para calcular y registrar similitudes basadas en coseno.  
  - `save_runs_simple_tag_shared_buffer.ipynb`: experimentos de novedad compartida con buffer compartido (se puede desactivar en la celda respectiva).

- **Paper Results/**  
  Contiene los resultados de los experimentos en formato `.csv` y el script para generar las figuras presentadas en el paper.  