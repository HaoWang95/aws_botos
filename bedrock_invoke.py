import boto3
import botocore
import json


amazon_titan_vite = "amazon.titan-text-lite-v1"
meta_llama2 = "meta.llama2-70b-chat-v1"
anthropic_claude2 = "anthropic.claude-v2"

accept_content = "application/json"

# default parameters for titan model
parameters = {"maxTokenCount": 2048, "stopSequences": [], "temperature": 0, "topP": 0.9}

amazon_titan_vite_map = {
    "modelId": "amazon.titan-text-lite-v1",
    "parameters": parameters,
}

bedrock_client = boto3.client("bedrock")
bedrock_runtime = boto3.client("bedrock-runtime")

request_label_duplicate_code = "Label the duplicate code snippet that should be removed:"
request_label_comment_code = "Write code comment for the snippet below:"


prompt_data = """
Question: write comment for the code below:
```python
from datetime import datetime

print(datetime.now())
```
Answer:
"""

react_prompt_data = """
Write code comment for the snippet below:

```TypeScript
import { PayloadAction, createSelector, createSlice } from "@reduxjs/toolkit";
import { CheckHealthResult } from "../api/taskApi";
import { RootState } from ".";

const initialState: CheckHealthResult = {
  statusCode: "",
  statusMessage: "",
};

export const serverStateSlice = createSlice({
  name: "serverState",
  initialState: initialState,
  reducers: {
    setServerState: (state, action: PayloadAction<CheckHealthResult>) => {
      state = action.payload;
    },
    setStatusCode: (state, action: PayloadAction<string>) => {
      state.statusCode = action.payload;
    },
    setStatusMessage: (state, action: PayloadAction<string>) => {
      state.statusMessage = action.payload;
    },
    setStatusCodeAction: (state, action: PayloadAction<string>) => {
      state.statusCode = action.payload;
    },
    setStatusMessageAction: (state, action) => {
      state.statusMessage = action.payload;
    },
  },
});

export const {
  setServerState,
  setStatusCode,
  setStatusMessage,
  setStatusCodeAction,
  setStatusMessageAction,
} = serverStateSlice.actions;

export default serverStateSlice.reducer;
```

"""

# default request body for titan model
request_body = json.dumps(
    {"inputText": react_prompt_data, "textGenerationConfig": parameters}
)

# default request body for meta llama2 model
meta_llama2_request_body_dup = json.dumps(
    {"prompt": react_prompt_data, "temperature": 0.5, "top_p": 0.9}
)

# default request body for anthropic claude2 model
anthropic_claude2_request_body = json.dumps(
    {"prompt": f"Human: {react_prompt_data}, Assistant:", "max_tokens_to_sample": 5000}
)


def call_model(bedrock_runtime, body, modelId, accept, contentType):
    print(f"Calling model {modelId}       " + "*"*100)
    try:
        response = bedrock_runtime.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        process_model_response(response=response, modelId=modelId)
    except botocore.exceptions.ClientError as err:
        print(err)
    print(f"Model call {modelId} finished " + "*"*100)


def process_model_response(response, modelId):
    response_body = json.loads(response.get("body").read())
    if modelId == amazon_titan_vite_map["modelId"]:
        print(response_body["results"][0]["outputText"])
    if modelId == meta_llama2:
        print(response_body["generation"])
    if modelId == anthropic_claude2:
        print(response_body["completion"])


if __name__ == "__main__":
    # invode amazon titan model
    call_model(
        bedrock_runtime=bedrock_runtime,
        body=request_body,
        modelId=amazon_titan_vite,
        accept=accept_content,
        contentType=accept_content,
    )

    print("*" * 100)
    call_model(
        bedrock_runtime=bedrock_runtime,
        body=meta_llama2_request_body_dup,
        modelId=meta_llama2,
        accept=accept_content,
        contentType=accept_content,
    )
    print("*" * 100)
    call_model(
        bedrock_runtime=bedrock_runtime,
        body=anthropic_claude2_request_body,
        modelId=anthropic_claude2,
        accept=accept_content,
        contentType=accept_content,
    )
