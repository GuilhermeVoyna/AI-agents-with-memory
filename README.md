# Lambda Function with Docker

Este repositório contém uma função AWS Lambda que utiliza Docker para fornecer um assistente de saúde baseado em IA. A função recebe dados do usuário e interage com APIs externas, como OpenAI e Qdrant, para fornecer respostas personalizadas.
## Primeiros passos

Clone o repositorio
```bash
  git clone https://github.com/GuilhermeVoyna/AI-agents-with-memory.git;
  cd AI-agents-with-memory
```
### Para editar lambda da aws localmente
Navegue para pasta image
```bash
   cd docker-lambda-aws/image
```
Build o docker
```bash
docker build -t lambda_with_docker . 
```
Inicie o docker
```bash
docker run -p 9000:8080 \
      -e OPENAI_API_KEY="sua_chave_openai" \
      -e QDRANT_API_KEY="sua_chave_qdrant" \
      -e QDRANT_API_URL="url_qdrant" \
      lambda_with_docker
```

#### Para usar com flask
Crie um ambiente virtual
```bash
   python3 -m .venv .venv
```
Instale os requerimentos
```bash
pip install -r requirements.txt
```
Inicie a main
```bash
python main.py
```
o suo do flask deve ser usado apenas para questao de teste/curiosidade
## Acessar api do aws local

#### Post message

```http
  POST /2015-03-31/functions/function/invocations
```
## Headers

| Header        | Value                          |
|---------------|--------------------------------|
| Content-Type  | application/json               |
| Authorization | Bearer {token}                |

## Request Body

O corpo da requisição deve ser um JSON com os seguintes parâmetros:

| Parameter           | Type      | Description                                         |
|---------------------|-----------|-----------------------------------------------------|
| `uid`               | `string`  | ID único do usuário.                               |
| `weight`            | `number`  | Peso do usuário em kg.                             |
| `height`            | `number`  | Altura do usuário em cm.                           |
| `gender`            | `string`  | Gênero do usuário (padrão: "Desconhecido").       |
| `message`           | `string`  | Mensagem a ser processada.                         |
| `bmi`               | `number`  | Índice de Massa Corporal calculado.               |
| `messages`          | `array`   | Lista de mensagens, contendo `content` e `role`.  |
| `birthday`          | `string`  | Data de nascimento do usuário em formato ISO.      |
| `exams_data`       | `object`  | Dados dos exames do usuário.                        |
| `appointments_data` | `object`  | Dados dos agendamentos do usuário.                 |
| `meds_data`        | `object`  | Dados sobre medicamentos do usuário.               |

### Example Request Body

```json
"{\"uid\":\"user123\",\"weight\":70,\"height\":175,\"gender\":\"Masculino\",\"message\":\"Como posso melhorar minha dieta?\",\"bmi\":22.86,\"messages\":[{\"content\":\"Qual é a sua altura?\",\"role\":\"assistant\"}],\"birthday\":\"1990-01-01T00:00:00.000Z\",\"exams_data\":{},\"appointments_data\":{},\"meds_data\":{}}"
```
### Para fazer testes pode usar o crul
```bash
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_TOKEN" \
-d "{\"uid\":\"user123\",\"weight\":70,\"height\":175,\"gender\":\"Masculino\",\"message\":\"Como posso melhorar minha dieta?\",\"bmi\":22.86,\"messages\":[{\"content\":\"Qual é a sua altura?\",\"role\":\"assistant\"}],\"birthday\":\"1990-01-01T00:00:00.000Z\",\"exams_data\":{},\"appointments_data\":{},\"meds_data\":{}}"
```
## Integração com o Mem0

Este projeto utiliza o [Mem0](https://github.com/mem0ai/mem0) para adicionar uma camada inteligente de memória ao chatbot. O Mem0 permite que o assistente de IA se lembre de interações passadas, preferências e dados do usuário, adaptando-se continuamente às necessidades do indivíduo. Essa integração eleva a personalização das respostas, melhorando a experiência do usuário a cada nova interação.

### Flexibilidade para Outras LLMs

Embora este exemplo utilize o OpenAI, o código pode ser facilmente adaptado para funcionar com outras LLMs, como **Llama**. Basta seguir a documentação do Mem0 e ajustar a integração conforme necessário.
