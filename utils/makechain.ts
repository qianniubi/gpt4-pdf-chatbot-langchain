// import { ChatOpenAI } from 'langchain/chat_models/openai';
import { ChatFireworks } from '@langchain/community/chat_models/fireworks';
import { ChatPromptTemplate } from 'langchain/prompts';
import { RunnableSequence } from 'langchain/schema/runnable';
import { StringOutputParser } from 'langchain/schema/output_parser';
import type { Document } from 'langchain/document';
import type { VectorStoreRetriever } from 'langchain/vectorstores/base';
import { BaseChatModel } from 'langchain/dist/chat_models/base';

const CONDENSE_TEMPLATE = `给出下面的对话和一个后续问题，将后续问题改写为一个独立的问题。

<chat_history>
  {chat_history}
</chat_history>

Follow Up Input: {question}
Standalone question:`;

const QA_TEMPLATE = `你是一位研究专家。使用以下上下文来回答最后的问题。 
如果你不知道答案，就说你不知道。不要试图编造答案。 
如果问题与上下文或聊天记录无关，请礼貌地回答您只会回答与上下文相关的问题。

<context>
  {context}
</context>

<chat_history>
  {chat_history}
</chat_history>

Question: {question}
请用简体中文回答:`;

const combineDocumentsFn = (docs: Document[], separator = '\n\n') => {
  const serializedDocs = docs.map((doc) => doc.pageContent);
  return serializedDocs.join(separator);
};

export const makeChain = (retriever: VectorStoreRetriever) => {
  const condenseQuestionPrompt =
    ChatPromptTemplate.fromTemplate(CONDENSE_TEMPLATE);
  const answerPrompt = ChatPromptTemplate.fromTemplate(QA_TEMPLATE);

  // const model = new ChatOpenAI({
  //   temperature: 0, // increase temperature to get more creative answers
  //   modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  // });

  const model = new ChatFireworks({
    fireworksApiKey: 'DAPyAOmfu9KHaAzHb6fj7obIAbcnn274rkSJEPzhjlkXjVjH',
    modelName: 'accounts/fireworks/models/mixtral-8x7b-instruct',
    temperature: 0,
    maxTokens: 20480
  }) as any as BaseChatModel;

  // Rephrase the initial question into a dereferenced standalone question based on
  // the chat history to allow effective vectorstore querying.
  const standaloneQuestionChain = RunnableSequence.from([
    condenseQuestionPrompt,
    model,
    new StringOutputParser(),
  ]);

  // Retrieve documents based on a query, then format them.
  const retrievalChain = retriever.pipe(combineDocumentsFn);

  // Generate an answer to the standalone question based on the chat history
  // and retrieved documents. Additionally, we return the source documents directly.
  const answerChain = RunnableSequence.from([
    {
      context: RunnableSequence.from([
        (input) => input.question,
        retrievalChain,
      ]),
      chat_history: (input) => input.chat_history,
      question: (input) => input.question,
    },
    answerPrompt,
    model,
    new StringOutputParser(),
  ]);

  // First generate a standalone question, then answer it based on
  // chat history and retrieved context documents.
  const conversationalRetrievalQAChain = RunnableSequence.from([
    {
      question: standaloneQuestionChain,
      chat_history: (input) => input.chat_history,
    },
    answerChain,
  ]);

  return conversationalRetrievalQAChain;
};
