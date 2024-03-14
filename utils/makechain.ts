// import { ChatOpenAI } from 'langchain/chat_models/openai';
import { ChatFireworks } from '@langchain/community/chat_models/fireworks';
import { ChatPromptTemplate } from 'langchain/prompts';
import { RunnableSequence } from 'langchain/schema/runnable';
import { StringOutputParser } from 'langchain/schema/output_parser';
import type { Document } from 'langchain/document';
import type { VectorStoreRetriever } from 'langchain/vectorstores/base';
import { BaseChatModel } from 'langchain/dist/chat_models/base';
import { mockChat } from './mock';

const CONDENSE_TEMPLATE = `给出下面的对话和一个后续问题，将后续问题改写为一个独立的问题。

<chat_history>
  {chat_history}
</chat_history>

问题是: {question}
Standalone Question:
`;

const QA_TEMPLATE = `你就是以为资深的心理咨询师。你收到的对话是一位患者的咨询，你需要回答患者的问题。请你回答患者的问题。
总结<chatcontent>和<chatcontent/>中的内容是类似的相关的聊天对话，你可以使用这些信息来回答问题。
如果问题和心理无关，你就回答：“嗯”，不要再说其他的了。
如果你不知道答案，就尝试去安慰一下，就回答：“我很能体会你的感受”。 
文字控制在50以下。
不要回答任何和心理学无关的内容。

<chatcontent>${mockChat}</chatcontent>

<context>
  {context}
</context>

对话是: {question}
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
      // chat_history: (input) => input.chat_history,
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
