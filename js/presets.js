/* TokenLab — sample training corpora and test sentences. */
(function (root) {
  'use strict';

  const corpora = [
    {
      name: 'Class example',
      text: [
        'The cat sat on the mat.',
        'The dog sat on the rug.',
        'The cat likes the dog.',
        'The sun is hot today.',
        'I like my cat and my dog.',
        'The dog ran to the park.',
      ].join('\n'),
    },
    {
      name: 'Bakery story',
      text:
        'Every morning the baker opened her small shop near the river. She baked bread, buns and biscuits ' +
        'before the sun was up. The baker\'s daughter helped her mix the flour and the water. They talked ' +
        'while they worked, and they laughed while they waited for the oven. When the bread was ready, the ' +
        'whole street smelled warm and sweet. Children walked past the window and waved. Workers stopped for ' +
        'a slice on their way to the station. By noon the shelves were empty, and the baker started planning ' +
        'tomorrow\'s baking. She wanted to try a new recipe: bread with seeds, honey and a little salt. Her ' +
        'daughter wanted to try baking cookies. They agreed to try both, and they baked until the evening.',
    },
    {
      name: 'Product reviews',
      text: [
        'Loved this phone! The camera is amazing and the battery lasts all day.',
        'The battery is good but the charger stopped working after a week.',
        'Amazing screen, amazing speakers, terrible packaging.',
        'I loved the colour but hated the slow delivery.',
        'Delivery was fast. Packaging was perfect. Loving it so far!',
        'The charger is slow, the camera is okay, the price is great.',
        'Worked perfectly for a month, then stopped charging. Disappointed.',
        'Great value. Great camera. Great battery. Would buy again!',
      ].join('\n'),
    },
    {
      name: 'Hindi + English',
      text: [
        'मुझे chai बहुत पसंद है। Mujhe chai bahut pasand hai.',
        'आज मौसम अच्छा है। Aaj mausam accha hai.',
        'मैं कल market जाऊँगा। Main kal market jaunga.',
        'यह phone बहुत अच्छा है। Yeh phone bahut accha hai.',
        'Chai aur samosa, best combo!',
      ].join('\n'),
    },
  ];

  const tests = [
    { name: 'Unknown word', text: 'The zebra sat on the mat.' },
    { name: 'New word forms', text: 'The cats were sitting on the rugs.' },
    { name: 'Mostly known', text: 'My dog likes the sunny park!' },
    { name: 'Strange characters', text: 'Café ☕ costs ₹40 — wow!' },
  ];

  root.Presets = { corpora, tests };
})(window);
