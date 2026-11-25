import express from "express"
import dotenv from "dotenv"
import fs from "fs"
import { ChatGoogle } from "@langchain/google";
import { ChatGroq } from "@langchain/groq"
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";


dotenv.config()
const app = express()
const port = 5000
app.use(express.json())


const llm = new ChatGroq({
    model: "openai/gpt-oss-120b",
    maxOutputTokens: 300,
    temperature: 1.5,
})

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
});


const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "resume-testing",
});


const upload = async () => {
    const pdfPath = "./updatedResume.pdf"
    const buffer = fs.readFileSync(pdfPath)
    const pdfResult = new PDFParse({ data: buffer })
    const textResult = await pdfResult.getText()
    const text = textResult.text
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 200, chunkOverlap: 50 })
    const docs =await splitter.createDocuments([text])
    await vectorStore.addDocuments(docs)
}


app.post("/ai", async (req, res) => {

    const { input } = req.body
    try {
        const docs=await vectorStore.similaritySearch(input,3)
        const llmcontext=docs.map((d)=>d.pageContent).join("/n")

        const response=await llm.invoke([
            new SystemMessage(`you are a RAG AI assistant.
                Rules:
                -answer ONLY from context
                -use ouside knowledge if crically nessacity arises
                - if answer not douns say : "I don,t know from uploaded PDF"
                Context : ${llmcontext}`),
                new HumanMessage(input)
        ])
        console.log(response)

        return res.status(200).json({
            ai:response.content
        })

    } catch (error) {
        console.error("Groq Error:", error);

        return res.status(500).json({
            message: "AI request failed"
        });
    }

})


app.get("/", (req, res) => {
    return res.json({
        message: "hello from level-1"
    })
})

app.listen(port, () => {
    console.log("server started")
})
