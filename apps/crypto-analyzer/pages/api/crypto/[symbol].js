export default async function handler(req, res) {
  const { symbol } = req.query;

  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  if (!symbol) {
    return res.status(400).json({ message: 'Symbol parameter required' });
  }

  try {
    const apiKey = process.env.COINGECKO_API_KEY || 'CG-htAfQpJ25kHiw7SxRNTjYNXA';
    const apiUrl = `https://api.coingecko.com/api/v3/coins/${symbol}/market_chart?vs_currency=usd&days=30&interval=daily`;

    const response = await fetch(apiUrl, {
      headers: {
        'x-cg-demo-api-key': apiKey
      }
    });

    if (!response.ok) {
      throw new Error(`Coingecko API error: ${response.status}`);
    }

    const data = await response.json();

    // Process the data
    const prices = data.prices.map(p => ({
      timestamp: p[0],
      date: new Date(p[0]).toISOString(),
      price: p[1]
    }));

    // Calculate momentum indicators
    const volatility = calculateVolatility(prices.map(p => p.price));
    const { adx, plusDI, minusDI } = calculateADX(prices.map(p => p.price));

    const latestPrice = prices[prices.length - 1];
    const previousPrice = prices[prices.length - 2] || latestPrice;
    const change = latestPrice.price - previousPrice.price;
    const changePercent = (change / previousPrice.price) * 100;

    res.status(200).json({
      success: true,
      symbol,
      latestPrice: latestPrice.price,
      change,
      changePercent,
      volatility,
      adx,
      plusDI,
      minusDI,
      prices: prices.slice(-10), // Last 10 days for charting
      fullData: prices
    });

  } catch (error) {
    console.error('Crypto API error:', error);

    // Fallback to mock data
    const mockData = generateMockCryptoData(symbol);
    res.status(200).json({
      success: true,
      symbol,
      isMock: true,
      ...mockData
    });
  }
}

function calculateVolatility(prices) {
  const returns = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
  return Math.sqrt(variance) * 100; // As percentage
}

function calculateADX(prices, period = 14) {
  if (prices.length < period * 2) return { adx: 0, plusDI: 0, minusDI: 0 };

  const highs = prices.map((p, i) => i > 0 ? Math.max(p, prices[i - 1]) : p);
  const lows = prices.map((p, i) => i > 0 ? Math.min(p, prices[i - 1]) : p);
  const closes = prices;

  const dmPlus = highs.map((h, i) => i > 0 ? Math.max(0, h - highs[i - 1]) : 0);
  const dmMinus = lows.map((l, i) => i > 0 ? Math.max(0, lows[i - 1] - l) : 0);
  const tr = closes.map((c, i) => i > 0 ? Math.max(
    Math.abs(c - lows[i - 1]),
    Math.abs(c - highs[i - 1]),
    Math.abs(highs[i - 1] - lows[i - 1])
  ) : 0);

  const atr = [];
  let atrSum = tr.slice(0, period).reduce((a, b) => a + b, 0);
  atr.push(atrSum / period);
  for (let i = period; i < tr.length; i++) {
    atrSum = atrSum - (atrSum / period) + tr[i];
    atr.push(atrSum / period);
  }

  const diPlus = [];
  const diMinus = [];
  for (let i = period; i < dmPlus.length; i++) {
    diPlus.push((dmPlus.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / atr[i - period]) * 100);
    diMinus.push((dmMinus.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / atr[i - period]) * 100);
  }

  const dx = diPlus.map((d, i) => Math.abs(d - diMinus[i]) / (d + diMinus[i]) * 100);
  const adx = dx.slice(0, period).reduce((a, b) => a + b, 0) / period;
  const plusDI = diPlus[diPlus.length - 1];
  const minusDI = diMinus[diMinus.length - 1];

  return { adx, plusDI, minusDI };
}

function generateMockCryptoData(symbol) {
  const startPrices = {
    bitcoin: 92000,
    ethereum: 2400,
    solana: 180,
    default: 5000
  };

  const basePrice = startPrices[symbol] || startPrices.default;
  const prices = [];
  let price = basePrice;

  for (let i = 0; i < 24; i++) {
    const change = (Math.random() - 0.5) * 0.1;
    price = price * (1 + change);
    prices.push({
      timestamp: Date.now() - (23 - i) * 3600000,
      date: new Date(Date.now() - (23 - i) * 3600000).toISOString(),
      price
    });
  }

  const volatility = calculateVolatility(prices.map(p => p.price));
  const { adx, plusDI, minusDI } = calculateADX(prices.map(p => p.price));

  const latestPrice = prices[prices.length - 1];
  const previousPrice = prices[prices.length - 2];
  const change = latestPrice.price - previousPrice.price;
  const changePercent = (change / previousPrice.price) * 100;

  return {
    latestPrice: latestPrice.price,
    change,
    changePercent,
    volatility,
    adx,
    plusDI,
    minusDI,
    prices: prices.slice(-10),
    fullData: prices
  };
}
