import type { JSX } from "react";
import { motion } from "framer-motion";

interface Spec {
  readonly title: string;
  readonly body: string;
  readonly fill: number;
}

interface PlaceholderProps {
  readonly heading: string;
  readonly accent: string;
  readonly blurb: string;
  readonly specs: readonly Spec[];
}

function Placeholder({ heading, accent, blurb, specs }: PlaceholderProps): JSX.Element {
  return (
    <div className="panel">
      <div className="panel-head">
        <h1>{heading} <em>{accent}</em></h1>
        <p>{blurb}</p>
      </div>
      <div className="engrave">planned instrumentation</div>
      <div className="cards">
        {specs.map((spec, i) => (
          <motion.div
            key={spec.title}
            className="card"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06, duration: 0.3 }}
          >
            <span className="tag-soon">phase</span>
            <h3>{spec.title}</h3>
            <p>{spec.body}</p>
            <div className="gauge"><i style={{ width: `${spec.fill}%` }} /></div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

export function WeightsPanel(): JSX.Element {
  return (
    <Placeholder
      heading="weight"
      accent="explorer"
      blurb="Real interpretability over the GGUF: per-layer tensor stats, the refusal direction as a first-class object, logit-lens projections, and activation patching."
      specs={[
        { title: "Tensor browser", body: "Per-layer norms, dtype, quant and shapes across the whole network.", fill: 16 },
        { title: "Logit lens", body: "Project intermediate residuals to vocabulary to read the model's mind mid-stack.", fill: 7 },
        { title: "Activation patching", body: "Causal tracing — patch activations to attribute behavior to components.", fill: 5 },
      ]}
    />
  );
}

