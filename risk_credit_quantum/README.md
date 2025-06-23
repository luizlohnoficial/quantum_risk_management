# Quantum Credit Risk Management

Projeto de exemplo para modelagem de risco de crédito utilizando recursos de computação quântica e clássica.

## Estrutura

```
risk_credit_quantum/
├── .github/workflows/ci-cd.yml
├── data/
├── notebooks/
├── src/
│   ├── api/
│   ├── dashboards/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── optimization/
│   └── simulation/
├── tests/
├── docs/
├── Dockerfile
└── requirements.txt
```

## Como executar

### Ambiente local

1. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Rode a API:
   ```bash
   uvicorn risk_credit_quantum.src.api.main:app --reload
   ```
3. Acesse `http://localhost:8000/docs` para ver a documentação interativa.

### Dashboard

```bash
streamlit run risk_credit_quantum.src.dashboards.app
```

### Testes

```bash
pytest
```

### Deploy em Nuvem

O projeto pode ser empacotado com Docker e enviado para Azure ou AWS. Utilize a pipeline de CI/CD em `.github/workflows/ci-cd.yml` para automatizar testes e build da imagem.

### Notebooks

Os notebooks de exploração ficam na pasta `notebooks/` e devem ser executados em ambiente que tenha Qiskit instalado.

## Governança

Consulte `docs/governance.md` para detalhes sobre aderência às normas Basileia III, IFRS9 e Bacen 3978.
