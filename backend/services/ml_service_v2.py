import pandas as pd
from typing import Optional, Dict, Any
from backend.models import RiskPrediction
from backend.features.build_features import build_features


class MLServiceV2:
    def __init__(self, model=None):
        self.model = model  # mockeado en tests / real en prod

    def predict(
        self,
        payload: Dict[str, Any],
        *,
        db=None,
    ) -> Dict[str, Any]:
        """
        Predice el riesgo para un solo registro.
        
        Args:
            payload: Diccionario con los datos del API
            db: Sesión de base de datos opcional para logging
            
        Returns:
            Diccionario con riesgo, confianza y probabilidades
        """
        # Convertir payload a DataFrame de una fila
        raw_df = pd.DataFrame([payload])
        
        # Usar predict_batch internamente
        result_df = self.predict_batch(
            raw_df,
            ciudad=payload.get("ciudad"),
            icao=payload.get("icao"),
            db=db,
        )
        
        # Retornar el primer resultado como diccionario
        return result_df.iloc[0].to_dict()

    def predict_batch(
        self,
        raw_df: pd.DataFrame,
        *,
        ciudad: Optional[str] = None,
        icao: Optional[str] = None,
        db=None,
    ) -> pd.DataFrame:
        X = build_features(raw_df)

        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        output = raw_df.copy()
        output["riesgo"] = preds
        output["confianza"] = probs.max(axis=1)

        if db is not None:
            for i in range(len(output)):
                record = RiskPrediction(
                    ciudad=ciudad,
                    icao=icao,
                    riesgo=str(preds[i]),
                    confianza=float(probs[i].max()),
                    probabilidades={
                        cls: float(p)
                        for cls, p in zip(self.model.classes_, probs[i])
                    },
                    temperatura=raw_df.iloc[i].get("temperatura"),
                    humedad=raw_df.iloc[i].get("humedad"),
                    viento=raw_df.iloc[i].get("viento"),
                    visibilidad=raw_df.iloc[i].get("visibilidad"),
                )
                db.add(record)

            db.commit()

        return output


ml_service_v2 = MLServiceV2()