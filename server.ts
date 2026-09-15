import express from "express";
import path from "path";
import dotenv from "dotenv";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";

dotenv.config();

const app = express();
const PORT = 3000;

// Set high body limit for handling document/PDF uploads in base64
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));

let geminiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!geminiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is required");
    }
    geminiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return geminiClient;
}

// Resilient helper to parse JSON responses from LLM, safely stripping markdown codeblocks
function safeJsonParse(text: string) {
  if (!text || !text.trim()) return {};
  const cleaned = text
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
  return JSON.parse(cleaned);
}

// Robust multi-model fallback caller
async function generateContentWithFallback(params: {
  contents: any;
  config?: any;
  preferredModel?: string;
}) {
  const ai = getGeminiClient();
  // Try preferred or standard flash first, then ultra-fast flash-lite, then latest alias
  const candidateModels = [
    params.preferredModel || "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
  ];

  let lastError: any = null;
  for (const model of candidateModels) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents: params.contents,
        config: params.config,
      });
      return response;
    } catch (err: any) {
      console.warn(`[Gemini API] Model ${model} encountered an issue, testing fallback...`, err?.message || err);
      lastError = err;
    }
  }

  // If all failed, extract human-readable error description
  let errMsg = lastError?.message || "Failed to generate content with Gemini AI.";
  try {
    const parsed = JSON.parse(errMsg);
    if (parsed?.error?.message) {
      errMsg = parsed.error.message;
    }
  } catch {
    // not stringified json
  }
  throw new Error(errMsg);
}

// Health check endpoint
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    hasApiKey: Boolean(process.env.GEMINI_API_KEY),
    time: new Date().toISOString(),
  });
});

// 1. PDF & Document Analysis / Synthesis Engine
app.post("/api/gemini/pdf-analyze", async (req, res) => {
  try {
    const { pdfBase64, textContent, mode, chapter, targetExam, count = 5 } = req.body;

    if (!pdfBase64 && (!textContent || !textContent.trim())) {
      return res.status(400).json({ error: "Please provide either a PDF document or text notes to analyze." });
    }

    let prompt = "";
    if (mode === "synthesize") {
      prompt = `You are "JEE Command AI", the premier question design engine for JEE Main & JEE Advanced.
Analyze the provided material (theory, formulas, questions, or chapter notes).
Generate ${count} fresh, original, high-yield JEE exam questions directly inspired by the concepts, formulas, or tricky traps found in this document.
Target Exam: ${targetExam || "JEE Advanced"}.
Focus Chapter / Topic: ${chapter || "General from document"}.

Strict Requirements:
1. Every math/physics/chemistry formula must use standard LaTeX: $...$ for inline, $$...$$ for block formulas.
2. Mix Question Types appropriate for ${targetExam}:
   - Single Choice (+4, -1)
   - Numerical / Integer (+4, 0)
   - Multi-Select (+4 partial, -2 incorrect) if Advanced
3. For each question provide:
   - Subject & Chapter
   - Question type
   - Formatted question statement with clear LaTeX
   - Options (A, B, C, D) if choice type
   - Correct answer (e.g. "B" or ["A", "C"] or "4.50")
   - In-depth, rigorous step-by-step solution
   - Key Concept flag
   - NCERT Mapping
   - PYQ Trend similarity (e.g. "Similar to JEE Advanced 2021 Paper 1")
   - Difficulty Level (Level 1: Concept Builder, Level 2: JEE Main Standard, Level 3: JEE Advanced Multi-Concept)`;
    } else if (mode === "extract") {
      prompt = `You are "JEE Command AI" Document Digitizer and OCR Processing Engine.
Your task is DIRECT EXTRACT MODE:
1. Digitally reproduce all questions found in the document/image.
2. Fix any typographical errors, scan imperfections, or OCR noise.
3. Re-format all mathematical equations, chemical formulas, and notations into clean standard LaTeX ($...$ inline, $$...$$ block).
4. Provide complete, rigorous, step-by-step pedagogical solutions for every extracted question with proven final answers.
5. If the document has fewer than 2 questions, extract all available questions and add high-yield complementary questions covering the core principles in the document.`;
    } else {
      // Concept summary & formulas
      prompt = `You are "JEE Command AI" High-Yield Theory & Formula Revision Engine.
Analyze the provided document (theory, formulas, notes, or problems) and produce a publication-grade "Revision Sheet":
# High-Yield Revision Digest
## 1. Executive Concept Summary & Core Axioms
(Crisp, high-density conceptual foundations)

## 2. Master Formula Sheet
(Every formula formatted in pristine LaTeX with variable explanations and SI units)

## 3. High-Yield "JEE Traps", Edge Cases & Boundary Conditions
(Common student misconceptions that cost -1 in JEE Main/Advanced)

## 4. Key Problem-Solving Shortcuts & Standard Results
(High-speed shortcuts for JEE Main and IIT JEE)`;
    }

    const contents: any[] = [];
    if (pdfBase64) {
      // Clean base64 string and extract mimeType
      let mimeType = "application/pdf";
      const match = String(pdfBase64).match(/^data:([^;]+);base64,/);
      if (match && match[1]) {
        mimeType = match[1];
      }
      const base64Clean = String(pdfBase64).includes(";base64,")
        ? String(pdfBase64).split(";base64,")[1]
        : String(pdfBase64).replace(/^data:[^,]+,/, "");

      contents.push({
        inlineData: {
          mimeType,
          data: base64Clean.trim(),
        },
      });
    }

    if (textContent && textContent.trim()) {
      contents.push({
        text: `Document text excerpt:\n"""\n${textContent.slice(0, 50000)}\n"""`,
      });
    }
    contents.push({ text: prompt });

    const isStructuredMode = mode === "synthesize" || mode === "extract";

    const response = await generateContentWithFallback({
      contents,
      config: {
        systemInstruction:
          "You are JEE Command AI, an authoritative, high-yield learning and document processing engine for JEE Main and JEE Advanced. Avoid conversational fluff. Use LaTeX everywhere appropriate. Ensure complete scientific and mathematical accuracy.",
        responseMimeType: isStructuredMode ? "application/json" : "text/plain",
        ...(isStructuredMode
          ? {
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  summary: { type: Type.STRING, description: "Executive overview or document synthesis summary" },
                  questions: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        id: { type: Type.STRING },
                        subject: { type: Type.STRING },
                        chapter: { type: Type.STRING },
                        type: { type: Type.STRING, description: "single_choice | numerical | multi_select" },
                        difficulty: { type: Type.STRING },
                        questionText: { type: Type.STRING },
                        options: {
                          type: Type.ARRAY,
                          items: { type: Type.STRING },
                        },
                        correctAnswer: { type: Type.STRING },
                        correctAnswers: {
                          type: Type.ARRAY,
                          items: { type: Type.STRING },
                        },
                        solution: { type: Type.STRING },
                        keyConcept: { type: Type.STRING },
                        ncertMapping: { type: Type.STRING },
                        pyqTrend: { type: Type.STRING },
                        marksCorrect: { type: Type.NUMBER },
                        marksIncorrect: { type: Type.NUMBER },
                      },
                      required: ["id", "subject", "chapter", "type", "questionText", "solution", "keyConcept"],
                    },
                  },
                },
                required: ["summary", "questions"],
              },
            }
          : {}),
      },
    });

    const text = response.text || "";
    if (isStructuredMode) {
      try {
        const parsed = safeJsonParse(text);
        res.json({ success: true, ...parsed });
      } catch (e) {
        res.json({ success: true, rawText: text });
      }
    } else {
      res.json({ success: true, text });
    }
  } catch (error: any) {
    console.error("PDF analyze error:", error);
    res.status(500).json({ error: error.message || "Failed to analyze document" });
  }
});

// 2. Mock Test Generator (JEE Main & JEE Advanced NTA Strict Patterns)
app.post("/api/gemini/generate-test", async (req, res) => {
  try {
    const {
      pattern = "jee-main",
      subject = "All",
      chapters = [],
      questionCount = 10,
      difficulty = "Standard",
    } = req.body;

    const prompt = `Generate a realistic, high-yield mock test paper adhering strictly to NTA and IIT JEE patterns.
Pattern: ${pattern === "jee-main" ? "JEE Main (Single Choice +4/-1 and Numerical Value +4/0)" : "JEE Advanced (Single Choice +4/-1, Multi-Correct +4 partial/-2, Integer/Numerical +4/0 or -1)"}.
Subject Focus: ${subject}
Selected Chapters: ${chapters.length > 0 ? chapters.join(", ") : "Standard High-Yield Mix"}
Total Questions: ${questionCount}
Difficulty Level: ${difficulty}

Rules:
1. Every question must be non-trivial, conceptual, and mathematically accurate.
2. Use precise LaTeX ($...$ and $$...$$) for all equations, formulas, vectors, integrals, chemical formulas.
3. Every question must have an authentic step-by-step pedagogical solution with the final numerical value or option clearly proven.
4. Include NCERT Mapping and Historical PYQ Trend tags (e.g., "Similar to JEE Advanced 2022 Paper 2, Q.14").
5. Provide standard marking parameters according to pattern.`;

    const response = await generateContentWithFallback({
      contents: prompt,
      config: {
        systemInstruction:
          "You are the NTA & IIT JEE Examination Setting Master. Produce impeccably formatted questions with flawless solutions.",
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            pattern: { type: Type.STRING },
            durationMinutes: { type: Type.INTEGER },
            totalMarks: { type: Type.INTEGER },
            instructions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            questions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  subject: { type: Type.STRING },
                  chapter: { type: Type.STRING },
                  type: { type: Type.STRING, description: "single_choice | numerical | multi_select" },
                  difficulty: { type: Type.STRING },
                  questionText: { type: Type.STRING },
                  options: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  correctAnswer: { type: Type.STRING, description: "Single option letter or numerical value string" },
                  correctAnswers: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: "For multi_select",
                  },
                  numericalTolerance: { type: Type.NUMBER },
                  solution: { type: Type.STRING },
                  keyConcept: { type: Type.STRING },
                  ncertMapping: { type: Type.STRING },
                  pyqTrend: { type: Type.STRING },
                  marksCorrect: { type: Type.NUMBER },
                  marksIncorrect: { type: Type.NUMBER },
                },
                required: [
                  "id",
                  "subject",
                  "chapter",
                  "type",
                  "questionText",
                  "solution",
                  "keyConcept",
                  "ncertMapping",
                  "marksCorrect",
                  "marksIncorrect",
                ],
              },
            },
          },
          required: ["title", "pattern", "durationMinutes", "totalMarks", "questions"],
        },
      },
    });

    const parsed = safeJsonParse(response.text || "{}");
    res.json(parsed);
  } catch (error: any) {
    console.error("Test generation error:", error);
    res.status(500).json({ error: error.message || "Failed to generate mock test" });
  }
});

// 2b. Automatic Gap Synthesizer (Fulfills exact shortlisting test length when question bank has gaps)
app.post("/api/gemini/gap-synthesize", async (req, res) => {
  try {
    const {
      pattern = "jee-advanced",
      subject = "Physics",
      chapter = "Rotational Motion",
      difficulty = "Level 3: JEE Advanced Multi-Concept",
      questionType = "all",
      count = 5,
      existingQuestionsSummary = [],
    } = req.body;

    const prompt = `You are the Master JEE Question Setter.
The student is assembling a custom test/DPP from their uploaded question library, but there is a gap of ${count} question(s) to fulfill the requested target criteria.

TARGET CRITERIA:
- Exam Pattern: ${pattern === "jee-main" ? "JEE Main Standard" : "JEE Advanced Strict Rigor"}
- Subject: ${subject}
- Chapter/Topic: ${chapter}
- Difficulty Level: ${difficulty}
- Required Format: ${questionType} (Options: single_choice, multi_select, numerical, matrix_match, or balanced mix if "all")
- Number of Fresh Original Questions Needed: ${count}

ALREADY SHORTLISTED (DO NOT DUPLICATE THESE):
${existingQuestionsSummary.length > 0 ? existingQuestionsSummary.slice(0, 8).join("; ") : "None. Generate distinct authentic high-yield problems."}

MANDATORY RULES:
1. Every question must be authentic, highly challenging, non-trivial, and mathematically pristine.
2. Format all equations and mathematical variables with LaTeX ($...$ inline, $$...$$ block).
3. If pattern is JEE Advanced, include multi-correct questions with partial marking (+4/-2) and numerical questions (+4/0), or matrix match (+4/-1).
4. For single_choice: marksCorrect = 4, marksIncorrect = 1.
5. For multi_select: marksCorrect = 4, marksIncorrect = 2.
6. Provide comprehensive pedagogical step-by-step solution derivations.
7. Include accurate NCERT mappings and PYQ Trend markers.`;

    const response = await generateContentWithFallback({
      contents: prompt,
      config: {
        systemInstruction:
          "You are an expert IIT JEE Problem Architect. Generate pristine, highly challenging, original JEE problems with rigorous LaTeX formatting.",
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            questions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  subject: { type: Type.STRING },
                  chapter: { type: Type.STRING },
                  type: {
                    type: Type.STRING,
                    description: "single_choice | multi_select | numerical | matrix_match",
                  },
                  difficulty: { type: Type.STRING },
                  questionText: { type: Type.STRING },
                  options: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  correctAnswer: { type: Type.STRING },
                  correctAnswers: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  solution: { type: Type.STRING },
                  keyConcept: { type: Type.STRING },
                  ncertMapping: { type: Type.STRING },
                  pyqTrend: { type: Type.STRING },
                  marksCorrect: { type: Type.NUMBER },
                  marksIncorrect: { type: Type.NUMBER },
                },
                required: [
                  "id",
                  "subject",
                  "chapter",
                  "type",
                  "questionText",
                  "solution",
                  "keyConcept",
                  "ncertMapping",
                  "marksCorrect",
                  "marksIncorrect",
                ],
              },
            },
          },
          required: ["questions"],
        },
      },
    });

    const parsed = safeJsonParse(response.text || "{}");
    const gapQuestions = (parsed.questions || []).map((q: any, i: number) => ({
      ...q,
      id: q.id || `synth_gap_${Date.now()}_${i}`,
      subject: q.subject || subject,
      chapter: q.chapter || chapter,
      origin: "gap_synthesized",
      timestamp: new Date().toISOString(),
    }));

    res.json({ success: true, questions: gapQuestions });
  } catch (error: any) {
    console.error("Gap synthesis error:", error);
    res.status(500).json({ error: error.message || "Failed to synthesize gap questions" });
  }
});

// 3. Adaptive DPP Engine
app.post("/api/gemini/generate-dpp", async (req, res) => {
  try {
    const {
      subject = "Physics",
      chapter = "Electrostatics",
      level = 2,
      count = 10,
      weakSpots = [],
      sourceContent = "",
    } = req.body;

    const levelDescriptions = {
      1: "Level 1: Concept Builder (Foundational theorems, direct formula applications, conceptual clarity)",
      2: "Level 2: JEE Main Standard (Moderate algebraic rigor, multi-step problem solving, standard traps)",
      3: "Level 3: JEE Advanced Multi-Concept (High cognitive load, boundary conditions, interlinked principles)",
    };

    const prompt = `Generate a targeted Daily Practice Problem (DPP) sheet.
Subject: ${subject}
Chapter: ${chapter}
Tier: ${levelDescriptions[level as keyof typeof levelDescriptions] || levelDescriptions[2]}
Number of Problems: ${count}
${weakSpots.length > 0 ? `Student Previous Weak Spots to prioritize: ${weakSpots.join(", ")}` : ""}
${sourceContent ? `Grounded in notes / document: ${sourceContent.slice(0, 2000)}` : ""}

Mandatory Output Criteria:
- Clear, unambiguous questions with LaTeX equations ($...$ and $$...$$).
- High diversity of sub-topics within ${chapter}.
- Detailed step-by-step working out in solution.
- NCERT section mapping & historical JEE relevance tag.`;

    const response = await generateContentWithFallback({
      contents: prompt,
      config: {
        systemInstruction:
          "You are JEE Command AI's Adaptive DPP Engine. Craft razor-sharp problems with pinpoint pedagogical accuracy.",
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            dppNumber: { type: Type.STRING },
            subject: { type: Type.STRING },
            chapter: { type: Type.STRING },
            targetLevel: { type: Type.STRING },
            learningObjectives: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            questions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  type: { type: Type.STRING, description: "single_choice | numerical | multi_select" },
                  questionText: { type: Type.STRING },
                  options: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  correctAnswer: { type: Type.STRING },
                  correctAnswers: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  solution: { type: Type.STRING },
                  keyConcept: { type: Type.STRING },
                  ncertMapping: { type: Type.STRING },
                  pyqTrend: { type: Type.STRING },
                  marksCorrect: { type: Type.NUMBER },
                  marksIncorrect: { type: Type.NUMBER },
                },
                required: ["id", "type", "questionText", "solution", "keyConcept"],
              },
            },
          },
          required: ["title", "subject", "chapter", "targetLevel", "questions"],
        },
      },
    });

    const parsed = safeJsonParse(response.text || "{}");
    res.json(parsed);
  } catch (error: any) {
    console.error("DPP generation error:", error);
    res.status(500).json({ error: error.message || "Failed to generate DPP" });
  }
});

// 4. Multi-Concept Fusion Generator
app.post("/api/gemini/multi-concept-fusion", async (req, res) => {
  try {
    const {
      chapters = ["Electrostatics", "Definite Integration & Calculus"],
      subject = "Physics & Mathematics",
      count = 3,
    } = req.body;

    const prompt = `You are the Master Question Setter for JEE Advanced.
Generate ${count} supreme "Multi-Concept Fusion" problems that interweave multiple chapters together into seamless, high-caliber JEE Advanced problems.
Target Chapters: ${chapters.join(" + ")}
Subject: ${subject}

Each problem must:
1. Genuinely require concepts from BOTH/ALL listed chapters to be solved (not just superficial mention).
2. For instance:
   - Electrostatics charge distribution coupled with differential equations/calculus
   - Thermodynamics work done integrated with organic reaction kinetics & equilibrium
   - Rotational dynamics linked with Simple Harmonic Motion and energy conservation
3. Provide:
   - "Concept Bridge": an explanation of how the two domains intersect.
   - Rigorous mathematical formulation in LaTeX.
   - Comprehensive multi-stage solution explaining the handoff between concept 1 and concept 2.
   - Pitfalls that catch 90% of JEE aspirants.`;

    const response = await generateContentWithFallback({
      contents: prompt,
      config: {
        systemInstruction:
          "You specialize in JEE Advanced Multi-Concept Fusion problems designed to separate top 500 rankers from the rest.",
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            theme: { type: Type.STRING },
            combinedChapters: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            questions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  title: { type: Type.STRING },
                  type: { type: Type.STRING, description: "single_choice | numerical | multi_select" },
                  questionText: { type: Type.STRING },
                  options: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  correctAnswer: { type: Type.STRING },
                  correctAnswers: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  conceptBridge: { type: Type.STRING, description: "How the two concepts unite in this problem" },
                  commonPitfall: { type: Type.STRING },
                  solution: { type: Type.STRING },
                  pyqTrend: { type: Type.STRING },
                },
                required: ["id", "title", "type", "questionText", "conceptBridge", "solution", "commonPitfall"],
              },
            },
          },
          required: ["theme", "combinedChapters", "questions"],
        },
      },
    });

    const parsed = safeJsonParse(response.text || "{}");
    res.json(parsed);
  } catch (error: any) {
    console.error("Multi-concept fusion error:", error);
    res.status(500).json({ error: error.message || "Failed to generate multi-concept fusion problem" });
  }
});

// 5. Socratic Doubt Resolver (Step-by-step guidance through hints first)
app.post("/api/gemini/socratic-doubt", async (req, res) => {
  try {
    const { questionContext, userQuery, hintLevel = 1, conversationHistory = [] } = req.body;

    const prompt = `You are the Socratic Doubt Resolver of JEE Command AI.
A student is stuck on this problem:
Problem Context:
"""
${questionContext}
"""

Student's Query or Attempt:
"""
${userQuery}
"""

Current Requested Hint Level: ${hintLevel} (1 = Conceptual Trigger / Fundamental Law, 2 = Setup & Diagram Guidance, 3 = Mathematical Key Step / Substitution, 4 = Full Reveal & Verification).

STRICT SOCRATIC PEDAGOGY RULE:
- DO NOT immediately provide the final numeric answer or direct option letter unless the student is explicitly on Hint Level 4!
- At Hint Level 1: Ask an incisive guiding question or point them to the governing physical law / mathematical theorem.
- At Hint Level 2: Help them draw the free-body diagram, coordinate frame, or identify the boundary conditions.
- At Hint Level 3: Show them how to set up the governing equation without executing the algebraic arithmetic for them.
- At Hint Level 4: Give the complete step-by-step derivation with final validation.
- Always use standard LaTeX ($...$) for mathematical symbols. Keep the tone encouraging, sharp, and mentor-like.`;

    const chatMessages: any[] = [];
    if (Array.isArray(conversationHistory)) {
      for (const msg of conversationHistory) {
        chatMessages.push({
          role: msg.role === "assistant" ? "model" : "user",
          parts: [{ text: msg.content }],
        });
      }
    }
    chatMessages.push({
      role: "user",
      parts: [{ text: prompt }],
    });

    const response = await generateContentWithFallback({
      contents: chatMessages,
      config: {
        systemInstruction:
          "You are a master JEE mentor who uses the Socratic method to build genuine conceptual mastery, never spoon-feeding answers prematurely.",
      },
    });

    res.json({
      success: true,
      hintLevel,
      reply: response.text || "Let's review the fundamental principles governing this setup.",
    });
  } catch (error: any) {
    console.error("Socratic doubt error:", error);
    res.status(500).json({ error: error.message || "Failed to resolve doubt" });
  }
});

// 6. Mistake Analysis & Digital Error Notebook Categorizer
app.post("/api/gemini/analyze-mistake", async (req, res) => {
  try {
    const {
      questionText,
      correctAnswer,
      userAnswer,
      timeTakenSeconds,
      averageBenchmarkSeconds = 120,
      userReasoning = "",
    } = req.body;

    const prompt = `Analyze this student error for the "Digital Error Notebook".
Question:
"""
${questionText}
"""
Correct Answer: ${correctAnswer}
Student's Answer: ${userAnswer || "Unattempted / Incorrect"}
Time Taken: ${timeTakenSeconds}s (Benchmark Average: ${averageBenchmarkSeconds}s)
Student Self-Explanation: "${userReasoning}"

Task:
1. Categorize root cause into exactly one of three core causes:
   - "Conceptual Gap" (Misunderstood fundamental definition, law, sign convention, or boundary condition)
   - "Calculation Error" (Concept was correct, but botched algebra, arithmetic, unit conversion, or power of 10)
   - "Time-Pressure Guess" (Rushed because timer was expiring, panic choice, eliminated improperly)
2. Determine Time vs. Accuracy status:
   - "Panic Guess" (if fast & wrong)
   - "Deep Confusion" (if slow & wrong)
   - "Calculative Slip" (if moderate time & arithmetic flaw)
3. Prescribe a 2-step Actionable Drill and NCERT Reference to permanently eliminate this error.`;

    const response = await generateContentWithFallback({
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            rootCause: {
              type: Type.STRING,
              description: "Conceptual Gap | Calculation Error | Time-Pressure Guess",
            },
            timeDiagnostic: {
              type: Type.STRING,
              description: "Panic Guess | Deep Confusion | Calculative Slip",
            },
            explanation: { type: Type.STRING },
            ncertChapter: { type: Type.STRING },
            drillRecommendation: { type: Type.STRING },
            keyFormulaToMemorize: { type: Type.STRING },
          },
          required: ["rootCause", "timeDiagnostic", "explanation", "drillRecommendation"],
        },
      },
    });

    const parsed = safeJsonParse(response.text || "{}");
    res.json(parsed);
  } catch (error: any) {
    console.error("Mistake analysis error:", error);
    res.status(500).json({ error: error.message || "Failed to analyze mistake" });
  }
});

// Explicit 404 for unhandled API endpoints - NEVER fall through to HTML SPA
app.all("/api/*", (req, res) => {
  res.status(404).json({ error: `API endpoint not found: ${req.method} ${req.originalUrl}` });
});

// Global Express error handling middleware - ALWAYS returns JSON
app.use((err: any, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error("Global Express error caught:", err);
  const status = err.status || err.statusCode || 500;
  res.status(status).json({
    error: err.message || "An internal server error occurred",
  });
});

// Vite middleware in dev, static serving in production
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`JEE Command AI Server active on http://0.0.0.0:${PORT}`);
  });
}

startServer();
