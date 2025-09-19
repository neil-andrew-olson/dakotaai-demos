const fs = require('fs');
const path = require('path');
constPapa = require('papaparse');

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    // Read the Titanic CSV file
    const csvPath = path.join(process.cwd(), 'titanic.csv');
    const csvData = fs.readFileSync(csvPath, 'utf8');

    // Parse CSV
    const parsed = Papa.parse(csvData, {
      header: true,
      dynamicTyping: true
    });

    if (parsed.errors.length > 0) {
      throw new Error('CSV parsing error: ' + JSON.stringify(parsed.errors));
    }

    const data = parsed.data;

    // Data cleaning and analysis
    const cleanedData = data.filter(passenger =>
      passenger.Pclass &&
      passenger.Survived !== undefined &&
      passenger.Age &&
      passenger.Fare &&
      passenger.Embarked
    );

    // Survival statistics
    const totalPassengers = cleanedData.length;
    const survived = cleanedData.filter(p => p.Survived === 1).length;
    const survivalRate = (survived / totalPassengers) * 100;

    // Class-based survival
    const byClass = [1, 2, 3].map(classNum => {
      const classP = cleanedData.filter(p => p.Pclass === classNum);
      const classSurvived = classP.filter(p => p.Survived === 1).length;
      return {
        class: classNum,
        count: classP.length,
        survived: classSurvived,
        survivalRate: classP.length > 0 ? (classSurvived / classP.length) * 100 : 0
      };
    });

    // Gender-based survival
    const bySex = ['male', 'female'].map(sex => {
      const sexP = cleanedData.filter(p => p.Sex === sex);
      const sexSurvived = sexP.filter(p => p.Survived === 1).length;
      return {
        sex,
        count: sexP.length,
        survived: sexSurvived,
        survivalRate: sexP.length > 0 ? (sexSurvived / sexP.length) * 100 : 0
      };
    });

    // Age groups
    const byAge = [
      { label: '0-18', min: 0, max: 18 },
      { label: '19-30', min: 19, max: 30 },
      { label: '31-50', min: 31, max: 50 },
      { label: '51+', min: 51, max: 100 }
    ].map(group => {
      const ageP = cleanedData.filter(p => p.Age >= group.min && p.Age <= group.max);
      const ageSurvived = ageP.filter(p => p.Survived === 1).length;
      return {
        ageGroup: group.label,
        count: ageP.length,
        survived: ageSurvived,
        survivalRate: ageP.length > 0 ? (ageSurvived / ageP.length) * 100 : 0
      };
    });

    // Fare distribution
    const fares = cleanedData.map(p => p.Fare).filter(f => f > 0);
    const avgFare = fares.reduce((a, b) => a + b, 0) / fares.length;
    const medianFare = [...fares].sort((a, b) => a - b)[Math.floor(fares.length / 2)];

    // Embarkation ports
    const embarkationGroups = cleanedData.reduce((acc, p) => {
      const port = acc.find(item => item.port === p.Embarked);
      if (port) {
        port.count++;
        if (p.Survived === 1) port.survived++;
        port.survivalRate = port.count > 0 ? (port.survived / port.count) * 100 : 0;
      } else {
        acc.push({
          port: p.Embarked,
          count: 1,
          survived: p.Survived === 1 ? 1 : 0,
          survivalRate: 0
        });
      }
      return acc;
    }, []);

    // Carried family
    const hasFamily = cleanedData.map(p => p.SibSp + p.Parch).filter(size => size > 0).length;
    const alone = cleanedData.length - hasFamily;

    res.status(200).json({
      success: true,
      summary: {
        totalPassengers: totalPassengers,
        survived: survived,
        survivalRate: survivalRate,
        processed: cleanedData.length,
        original: data.length
      },
      byClass,
      bySex,
      byAge,
      embarkation: embarkationGroups,
      demographics: {
        avgAge: cleanedData.reduce((sum, p) => sum + p.Age, 0) / cleanedData.length,
        avgFare: avgFare,
        medianFare: medianFare,
        family: {
          withFamily: hasFamily,
          alone: alone
        }
      },
      insights: [
        {
          title: "Survival by Socioeconomic Class",
          description: "First-class passengers had significantly higher survival rates than third-class passengers.",
          data: byClass
        },
        {
          title: "Gender Impact on Survival",
          description: "Women had much higher survival rates than men, reflecting the 'women and children first' policy.",
          data: bySex
        },
        {
          title: "Age and Vulnerability",
          description: "Children under 18 had better survival chances, while adults in middle age faced lowest survival rates.",
          data: byAge
        }
      ]
    });

  } catch (error) {
    console.error('Titanic analysis error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to analyze Titanic data',
      error: error.message
    });
  }
}
