import { NODE_CATALOG, NODE_CATEGORIES, catalogEntriesByCategory } from "../nodeCatalog";

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string) {
  if (condition) passed++;
  else {
    failed++;
    console.error(`FAIL: ${message}`);
  }
}

// Every catalog entry has a category that's one of the known categories.
{
  const known = new Set(NODE_CATEGORIES);
  const bad = NODE_CATALOG.filter((e) => !known.has(e.category));
  assert(bad.length === 0, `all entries have a known category — offenders: ${bad.map((e) => e.type).join(", ")}`);
}

// No duplicate node types in the catalog.
{
  const types = NODE_CATALOG.map((e) => e.type);
  const unique = new Set(types);
  assert(unique.size === types.length, `no duplicate types in the catalog — had ${types.length}, ${unique.size} unique`);
}

// Every category referenced in NODE_CATEGORIES has at least one entry (no empty section in the palette).
{
  for (const category of NODE_CATEGORIES) {
    const entries = catalogEntriesByCategory(category);
    assert(entries.length > 0, `category '${category}' has at least one catalog entry`);
  }
}

// Probabilistic-flagged entries are only AI-category node types (matches the compiler's own PROBABILISTIC_CAPABLE_TYPES constraint in agent_spec.py).
{
  const wronglyProbabilistic = NODE_CATALOG.filter((e) => e.defaultExecutionType === "probabilistic" && e.category !== "ai");
  assert(
    wronglyProbabilistic.length === 0,
    `only 'ai' category entries default to probabilistic — offenders: ${wronglyProbabilistic.map((e) => e.type).join(", ")}`
  );
}

// A reasonable minimum catalog size (regression guard against an accidental near-empty catalog).
assert(NODE_CATALOG.length >= 25, `catalog covers a meaningful range of node types — got ${NODE_CATALOG.length}`);

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
