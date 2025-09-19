import { useState, useEffect } from 'react';

export default function TitanicAnalyzer() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAnalysis();
  }, []);

  const fetchAnalysis = async () => {
    try {
      const response = await fetch('/api/titanic/analyze');
      if (!response.ok) throw new Error('Failed to fetch analysis');
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  if (loading) {
    return (
      <div style={{ padding: 20, textAlign: 'center' }}>
        <h1>Titanic Dataset Explorer</h1>
        <p>Analyzing passenger data...</p>
        <div style={{ width: 40, height: 40, border: '4px solid #f3f4f6', borderTop: '4px solid #2563eb', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '20px auto' }}></div>
        <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: 20, textAlign: 'center' }}>
        <h1>Error</h1>
        <p>{error}</p>
        <button onClick={fetchAnalysis}>Try Again</button>
      </div>
    );
  }

  if (!data) return null;

  const { summary, byClass, bySex, byAge, demographics, insights } = data;

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: 20, fontFamily: 'Arial, sans-serif' }}>
      <h1 style={{ textAlign: 'center', color: '#1e293b' }}>Titanic Dataset Explorer</h1>
      <p style={{ textAlign: 'center', color: '#64748b', maxWidth: 800, margin: '0 auto' }}>
        Machine Learning & Statistical Analysis of Titanic Passenger Survival Data
      </p>

      <div style={{ background: 'white', padding: 20, borderRadius: 8, boxShadow: '0 4px 12px rgba(0,0,0,0.1)', margin: '20px 0' }}>
        <h2>Overview Statistics</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 20 }}>
          <div style={{ textAlign: 'center', padding: 20, background: '#f8fafc', borderRadius: 8 }}>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#2563eb' }}>{summary.totalPassengers}</div>
            <div style={{ color: '#64748b' }}>Total Passengers</div>
          </div>
          <div style={{ textAlign: 'center', padding: 20, background: '#f8fafc', borderRadius: 8 }}>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#10b981' }}>{summary.survived}</div>
            <div style={{ color: '#64748b' }}>Survived</div>
          </div>
          <div style={{ textAlign: 'center', padding: 20, background: '#f8fafc', borderRadius: 8 }}>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#f59e0b' }}>{summary.survivalRate.toFixed(1)}%</div>
            <div style={{ color: '#64748b' }}>Survival Rate</div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 20 }}>
        <div style={{ background: 'white', padding: 20, borderRadius: 8, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
          <h3>Survival by Passenger Class</h3>
          {byClass.map(cls => (
            <div key={cls.class} style={{ margin: '10px 0', padding: 10, background: '#f8fafc', borderRadius: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Class {cls.class}</span>
                <span style={{ fontWeight: 'bold', color: '#2563eb' }}>{cls.survivalRate.toFixed(1)}% ({cls.survived}/{cls.count})</span>
              </div>
              <div style={{ width: '100%', height: 8, background: '#e5e7eb', borderRadius: 4, marginTop: 5 }}>
                <div style={{ width: `${cls.survivalRate}%`, height: '100%', background: '#2563eb', borderRadius: 4, transition: 'width 0.5s' }}></div>
              </div>
            </div>
          ))}
        </div>

        <div style={{ background: 'white', padding: 20, borderRadius: 8, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
          <h3>Survival by Gender</h3>
          {bySex.map(sex => (
            <div key={sex.sex} style={{ margin: '10px 0', padding: 10, background: '#f8fafc', borderRadius: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ textTransform: 'capitalize' }}>{sex.sex}</span>
                <span style={{ fontWeight: 'bold', color: '#2563eb' }}>{sex.survivalRate.toFixed(1)}%</span>
              </div>
              <div style={{ width: '100%', height: 8, background: '#e5e7eb', borderRadius: 4, marginTop: 5 }}>
                <div style={{ width: `${sex.survivalRate}%`, height: '100%', background: '#dc2626', borderRadius: 4, transition: 'width 0.5s' }}></div>
              </div>
            </div>
          ))}
        </div>

        <div style={{ background: 'white', padding: 20, borderRadius: 8, boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
          <h3>Survival by Age Group</h3>
          {byAge.map(age => (
            <div key={age.ageGroup} style={{ margin: '10px 0', padding: 10, background: '#f8fafc', borderRadius: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>{age.ageGroup} years</span>
                <span style={{ fontWeight: 'bold', color: '#2563eb' }}>{age.survivalRate.toFixed(1)}%</span>
              </div>
              <div style={{ width: '100%', height: 8, background: '#e5e7eb', borderRadius: 4, marginTop: 5 }}>
                <div style={{ width: `${age.survivalRate}%`, height: '100%', background: '#10b981', borderRadius: 4, transition: 'width 0.5s' }}></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ background: 'white', padding: 20, borderRadius: 8, boxShadow: '0 4px 12px rgba(0,0,0,0.1)', marginTop: 20 }}>
        <h3>Demographic Insights</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 20 }}>
          <div style={{ padding: 15, background: '#f8fafc', borderRadius: 6 }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#2563eb' }}>${demographics.avgFare.toFixed(2)}</div>
            <div style={{ color: '#64748b' }}>Average Fare</div>
          </div>
          <div style={{ padding: 15, background: '#f8fafc', borderRadius: 6 }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#2563eb' }}>{demographics.avgAge.toFixed(1)}</div>
            <div style={{ color: '#64748b' }}>Average Age</div>
          </div>
          <div style={{ padding: 15, background: '#f8fafc', borderRadius: 6 }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#10b981' }}>{demographics.family.withFamily}</div>
            <div style={{ color: '#64748b' }}>Travelled with Family</div>
          </div>
          <div style={{ padding: 15, background: '#f8fafc', borderRadius: 6 }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#dc2626' }}>{demographics.family.alone}</div>
            <div style={{ color: '#64748b' }}>Travelled Alone</div>
          </div>
        </div>
      </div>

      <div style={{ background: 'white', padding: 20, borderRadius: 8, boxShadow: '0 4px 12px rgba(0,0,0,0.1)', marginTop: 20 }}>
        <h3>Key Findings</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 20 }}>
          {insights.map((insight, index) => (
            <div key={index} style={{ padding: 15, border: '1px solid #e5e7eb', borderRadius: 6 }}>
              <h4 style={{ margin: '0 0 10px 0', color: '#1e293b' }}>{insight.title}</h4>
              <p style={{ color: '#64748b', margin: 0 }}>{insight.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
