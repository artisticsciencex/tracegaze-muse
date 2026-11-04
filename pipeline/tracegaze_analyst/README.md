# TraceGaze analyst v1

This is the minimal evidence-grounded LLM layer. It does four things:

1. converts a validated attention trace into deterministic, citable facts;
2. selects a bounded evidence packet for a researcher question;
3. requires an LLM response to separate measured facts, observed patterns,
   interpretations, hypotheses and limitations;
4. rejects unknown evidence references and renders a report for human checking.

Model use is optional. Packet construction, deterministic facts, response
validation, and report rendering do not depend on a model provider. A local
Ollama adapter is included for semantic analysis without paid API use or data
upload; externally obtained response JSON can be validated through the same
interface.

```bash
cd pipeline
python -m tracegaze_analyst prepare \
  --trace /path/to/attention_trace \
  --question "Which objects structured this session?" \
  --out /tmp/question_packet

python -m tracegaze_analyst baseline \
  --packet /tmp/question_packet/evidence_packet.json \
  --out /tmp/question_report

# After obtaining JSON from a chosen model:
python -m tracegaze_analyst validate \
  --packet /tmp/question_packet/evidence_packet.json \
  --response /path/to/model_response.json \
  --out /tmp/validated_report
```

The baseline is not presented as semantic analysis; it is a zero-API integration
test. The local provider adapter was exercised only after the deterministic path
passed on both NEON and Aria data.



## Local semantic baseline

```bash
python -m tracegaze_analyst analyze-local \
  --packet /tmp/question_packet/evidence_packet.json \
  --model qwen3.6:35b \
  --out /tmp/local_analysis
```

This uses the local Ollama service. It makes no paid API call and uploads no data.
