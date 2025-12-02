**Your Idea: Consultation**

```
Claude stuck → Ask GPT-4 "What am I missing?" → Claude uses insight
```

Collaborative, additive.

---

## Why This Works

**LLMs Have Different Strengths**

<pre class="font-ui border-border-100/50 overflow-x-scroll w-full rounded border-[0.5px] shadow-[0_2px_12px_hsl(var(--always-black)/5%)]"><table class="bg-bg-100 min-w-full border-separate border-spacing-0 text-sm leading-[1.88888] whitespace-normal"><thead class="border-b-border-100/50 border-b-[0.5px] text-left"><tr class="[tbody>&]:odd:bg-bg-500/10"><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] px-2 [&:not(:first-child)]:border-l-[0.5px]">Aspect</th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] px-2 [&:not(:first-child)]:border-l-[0.5px]">Claude</th><th class="text-text-000 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] px-2 [&:not(:first-child)]:border-l-[0.5px]">GPT-4</th></tr></thead><tbody><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Reasoning</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Strong</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Strong</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Code style</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Clean, verbose</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Compact, pragmatic</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Edge cases</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Cautious</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Risk-taking</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Context handling</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Large window</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Better compression</td></tr><tr class="[tbody>&]:odd:bg-bg-500/10"><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Blind spots</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Different</td><td class="border-t-border-100/50 [&:not(:first-child)]:-x-[hsla(var(--border-100) / 0.5)] border-t-[0.5px] px-2 [&:not(:first-child)]:border-l-[0.5px]">Different</td></tr></tbody></table></pre>

**Key Insight**

When stuck, the problem is often framing or blind spot. Different LLM sees differently.

---

## Consultation Patterns

**Pattern 1: Second Opinion**

```
Claude generates code
Tests fail 2x
        ↓
Yantra: "Claude, here's the code. Here's the error.
         GPT-4, what's Claude missing?"
        ↓
GPT-4: "The issue is X. Try approach Y."
        ↓
Yantra: "Claude, consider this: [GPT-4's insight].
         Now fix the code."
        ↓
Claude fixes with new perspective
```

**Pattern 2: Alternative Approach**

```
Claude stuck on implementation
        ↓
Yantra: "GPT-4, how would YOU implement this?"
        ↓
GPT-4: Returns different implementation
        ↓
Yantra: "Claude, here's an alternative approach.
         Adapt or merge with your solution."
        ↓
Claude synthesizes best of both
```

**Pattern 3: Debug Partner**

```
Claude's code has subtle bug
Claude can't find it after 2 attempts
        ↓
Yantra: "GPT-4, review this code. Find the bug."
        ↓
GPT-4: "Line 47: Race condition when X happens"
        ↓
Yantra: "Claude, fix the race condition on line 47."
        ↓
Claude fixes specific issue
```

**Pattern 4: Validation**

```
Claude generates complex solution
        ↓
Yantra: "GPT-4, any issues with this approach?"
        ↓
GPT-4: "Edge case: What if Y is null?"
        ↓
Yantra: "Claude, handle the null case for Y."
        ↓
More robust solution
```

---

## Implementation

**Consulting Agent**

rust

```rust
structConsultingAgent{
    primary:LlmClient,// Claude
    consultant:LlmClient,// GPT-4
    consultation_threshold:u32,// After N failures
}

implConsultingAgent{
asyncfngenerate_with_consultation(
&self,
        task:&Task,
)->Result<Code>{
letmut attempts =0;
letmut context =ConsultationContext::new();
      
loop{
// Try primary LLM
let result =self.primary.generate(task,&context).await?;
let test_result =self.test(&result).await?;
          
if test_result.passed {
returnOk(result);
}
          
            attempts +=1;
            context.add_failure(result, test_result);
          
// Consult after threshold
if attempts >=self.consultation_threshold {
let insight =self.consult(&context).await?;
                context.add_insight(insight);
              
// Reset attempts, try with new insight
                attempts =0;
}
          
if context.total_attempts()>6{
returnErr(Error::NeedsHuman);
}
}
}
  
asyncfnconsult(&self, context:&ConsultationContext)->Result<Insight>{
let prompt =format!(
"Another AI attempted this task and failed.
           
             Task: {}
           
             Attempts:
             {}
           
             Errors:
             {}
           
             What is being missed? Suggest a different approach.",
            context.task,
            context.attempts_summary(),
            context.errors_summary(),
);
      
self.consultant.generate(&prompt).await
}
}
```

---

## ConsultationPromptTemplates

**SecondOpinionPrompt**

```
I'm working on this task:{task_description}

I tried this approach:
```

{failed_code}

```

But got this error:
```

{error_message}

```

I've tried {N} times with similar results.

What am I missing?What's a different way to think about this?
```

**DebugPartnerPrompt**

```
Review this code for bugs:
```

{code}

```

The tests are failing with:
```

{test_failures}

```

The original developer couldn't find the issue.
What's wrong?
```

**AlternativeApproachPrompt**

```
Task:{task_description}

Current implementation approach:
{current_approach_summary}

This approach is hitting issues with:
{issues}

How would you implement this differently?
Provide a complete alternative solution.
```

---

## WhenToConsult

| Situation                   | Action                        |
| --------------------------- | ----------------------------- |
| First failure               | Retry with same LLM           |
| Second failure              | Retry with refined prompt     |
| Third failure               | Consult other LLM             |
| Fourthfailure(with insight) | Try with consultation insight |
| Fifth failure               | Try other LLMas primary       |
| Sixth failure               | Escalate to human             |

---

## CostConsideration

**Consultation adds cost, but...**

Without consultation:

```
Attempt1:Claude → Fail
Attempt2:Claude → Fail  
Attempt3:Claude → Fail
Attempt4:Claude → Fail
Attempt5:Claude → Fail
→ Human intervention needed
Total:5LLM calls, no solution
```

With consultation:

```
Attempt1:Claude → Fail
Attempt2:Claude → Fail
Consult:GPT-4  → Insight
Attempt3:Claude → Success
Total:4LLM calls, solved
```

**Consultation often cheaper than repeated failures.**

---

## TrackingEffectiveness

**Metrics to Capture**

```
-Consultations triggered:234/month
-Consultations that helped:189(81%)
-Avg attempts before consultation:2.3
-Avg attempts after consultation:1.4
-Cost per consultation:$0.08
-Costsaved(vs human intervention):$12/incident
```

**InsightPatterns(forCodex later)**

```
Store successful consultations:
{
"task_type":"async_error_handling",
"primary_blind_spot":"missing await",
"consultant_insight":"check all async functions have await",
"success":true
}
```

---

## Extended:Three-WayConsultation

**ForReallyHardProblems**

```
Claude fails 3x
GPT-4 consulted, insight doesn't help
        ↓
Yantra:"Gemini, both Claude and GPT-4 struggled.
         Here's what they tried. Fresh perspective?"
        ↓
Gemini provides third approach
        ↓
Claude synthesizes all three
```

**LLMRoster**

| Role        | Model              | Strength                         |
| ----------- | ------------------ | -------------------------------- |
| Primary     | Claude             | Reasoning, safety                |
| Consultant1 | GPT-4              | Pragmatic, different perspective |
| Consultant2 | Gemini             | Fresh take, different training   |
| Specialist  | Codestral/DeepSeek | Code-specific                    |

---

## UITransparency

**ShowUserWhat'sHappening**

```
┌─────────────────────────────────────────────────────────────┐
│ Generating:User authentication                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ✅ Attempt1:Generated auth module                         │
│ ❌ Tests failed:JWT validation error                       │
│                                                             │
│ ✅ Attempt2:FixedJWT validation                          │
│ ❌ Tests failed:Session handling edge case                 │
│                                                             │
│ 🤔 Consulting second opinion...                             │
│ 💡 Insight:"Session needs refresh token rotation"          │
│                                                             │
│ ✅ Attempt3:Added refresh token rotation                  │
│ ✅ All tests passing                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**User sees:** Yantra tried, got stuck, asked for help, solved it.

**User feels:** Thorough, not just blind retrying.
