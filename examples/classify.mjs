// Support-ticket triage with the JavaScript client (Node 18+). Reads CLEF_URL and CLEF_API_KEY.
import { ClefClient } from '../clients/js/src/index.js';

const TICKET = 'Checkout is down, orders blocked';
const LABELS = ['billing', 'technical', 'account'];

const clef = new ClefClient(); // baseUrl / apiKey default to CLEF_URL / CLEF_API_KEY

// Single label
let r = await clef.classify(TICKET, LABELS);
console.log(`single : ${r.label} (${r.confidence.toFixed(2)})`, r.scores);

// Multi label: every label whose score is >= threshold, best first
r = await clef.classify(
  'I was charged twice and the app crashes on login',
  {
    billing: 'Payments, invoices, refunds',
    technical: 'Bugs and outages',
    account: 'Login, profile, permissions',
  },
  { multiLabel: true, threshold: 0.5 },
);
console.log('multi  :', r.labels, r.scores);

// Ordinal score
const s = await clef.score(TICKET, ['low', 'medium', 'high'], { instructions: 'How urgent is this ticket?' });
console.log(`score  : ${s.level} (score=${s.score.toFixed(2)})`);

// Many inputs in one request
const texts = ['Refund my invoice', 'Cannot reset my password'];
(await clef.classifyMany(texts, LABELS)).forEach((res, i) => console.log(`batch  : ${texts[i]} -> ${res.label}`));

// Saved classifier
const triage = clef.classifier('support-triage');
await triage.save({ labels: LABELS, description: 'Support ticket routing' });
console.log('saved  :', (await triage.classify(TICKET)).label);
await triage.delete();
