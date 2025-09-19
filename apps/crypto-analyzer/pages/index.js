import { useState, useEffect } from 'react';

export default function CryptoAnalyzer() {
  const [symbol, setSymbol] = useState('bitcoin');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prices, setPrices] = useState([]);

  const fetchCryptoData = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/crypto/${symbol}`);
      const result = await response.json();
      setData(result);
      setPrices(result.prices || []);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
    setLoading(false);
  };

  const getMomentumAnalysis = (adxVal, plusDI, minusDI) => {
    let strength = 'Weak Trend (ADX < 20)';
    if (adxVal >= 25) strength = 'Strong Trend (ADX >= 25)';
    else if (adxVal >= 20) strength = 'Moderate Trend (20 <= ADX < 25)';
    let direction = 'Neutral';
    if (plusDI > minusDI) direction = 'Bullish (+DI > -DI)';
    else if (minusDI > plusDI) direction = 'Bearish (-DI > +DI)';
    return { strength, direction };
  };

  useEffect(() => {
    fetchCryptoData();
  }, []);

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', margin: 0, padding: 20, backgroundColor: '#f5f5f5' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', background: 'white', padding: 20, borderRadius: 8, boxShadow: '0 0 10px rgba(0,0,0,0.1)' }}>
        <h1 style={{ color: '#333', textAlign: 'center' }}>Dakota AI Demo: Crypto Momentum Analyzer</h1>
        <p>This demo analyzes cryptocurrency trends, volatility, and momentum indicators for short-term bullish/bearish/neutral analysis. Enter a crypto symbol (e.g., bitcoin) to process data and visualize insights.</p>
        <input
          type="text"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toLowerCase())}
          placeholder="Enter Crypto Symbol"
          style={{ display: 'block', margin: '20px auto', padding: 10, width: 200 }}
        />
        <button onClick={fetchCryptoData} disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
        {data && (
          <div id="cryptoData" style={{ marginTop: 20 }}>
            <h2>{symbol.toUpperCase()} - Market Analysis {data.isMock ? '(Mock Data)' : ''}</h2>
            <p><strong>Latest Price:</strong> ${data.latestPrice?.toFixed(2)}</p>
            <p><strong>Change:</strong> ${data.change?.toFixed(2)} ({data.changePercent?.toFixed(2)}%)</p>
            <p><strong>Volatility (1-day):</strong> {data.volatility?.toFixed(2)}%</p>
            <p><strong>ADX:</strong> {data.adx?.toFixed(2)}</p>
            <p><strong>Trend Strength:</strong> {getMomentumAnalysis(data.adx, data.plusDI, data.minusDI).strength}</p>
            <p><strong>Momentum Direction:</strong> {getMomentumAnalysis(data.adx, data.plusDI, data.minusDI).direction}</p>
            <p><strong>+DI:</strong> {data.plusDI?.toFixed(2)} | <strong>-DI:</strong> {data.minusDI?.toFixed(2)}</p>
          </div>
        )}
        <canvas id="cryptoChart" style={{ display: 'block', margin: '20px auto', width: '80vw', height: '50vh', maxWidth: 800, maxHeight: 400 }}></canvas>
      </div>
    </div>
  );
}
