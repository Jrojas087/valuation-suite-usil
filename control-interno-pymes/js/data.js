// Course content and quiz data for Control Interno PYMES Paraguay
const COURSE = {
  title: "Control Interno para PYMES",
  subtitle: "Curso de Auto-instrucción",
  institution: "Programa de Capacitación Empresarial — Paraguay",
  passingScore: 70,

  modules: [
    {
      id: 1,
      title: "Fundamentos del Control Interno",
      shortTitle: "Fundamentos",
      duration: "25 min",
      icon: "📚",
      content: `
        <h2>¿Qué es el Control Interno?</h2>
        <p>El <strong>control interno</strong> es el conjunto de reglas, hábitos y revisiones que una empresa pone en práctica para cuidar su dinero, sus bienes y su reputación. Lo aplican el dueño, los gerentes y todos los empleados — no es tarea de una sola persona.</p>
        <p>Su propósito es dar <em>certeza razonable</em> (no absoluta) de que la empresa logra tres tipos de objetivos:</p>
        <ul>
          <li><strong>Objetivos operativos:</strong> Que las cosas se hagan bien y sin desperdiciar recursos.</li>
          <li><strong>Objetivos de información:</strong> Que los datos que usa la empresa —ventas, costos, inventarios— sean confiables.</li>
          <li><strong>Objetivos de cumplimiento:</strong> Que la empresa cumpla las leyes y regulaciones, como las de la DNIT.</li>
        </ul>
        <div class="highlight-box">
          <strong>Punto clave:</strong> Ningún sistema de control elimina todos los riesgos. Siempre queda algo de incertidumbre porque trabajan personas y hay situaciones imprevistas. Por eso decimos que da "certeza razonable", no garantía total.
        </div>
        <h2>El Marco COSO — la referencia mundial</h2>
        <p>El <strong>Marco COSO 2013</strong> (siglas en inglés de un comité de organizaciones profesionales) es el sistema de referencia más usado en el mundo para diseñar y evaluar el control interno. Define <strong>cinco componentes</strong> que toda organización debería tener:</p>
        <ol>
          <li>Entorno de Control</li>
          <li>Evaluación de Riesgos</li>
          <li>Actividades de Control</li>
          <li>Información y Comunicación</li>
          <li>Actividades de Supervisión</li>
        </ol>
        <p>A lo largo de este curso vamos a ver cada uno en detalle, con ejemplos prácticos para PYMES paraguayas.</p>
        <h2>¿Por qué importa para las PYMES en Paraguay?</h2>
        <p>Las PYMES representan más del <strong>90 % del tejido empresarial paraguayo</strong>. Sin controles básicos, muchas enfrentan problemas que se podrían evitar:</p>
        <ul>
          <li>Pérdidas por robo o errores que nadie detecta a tiempo</li>
          <li>Dificultades para conseguir crédito en el BNF o la AFD</li>
          <li>Multas de la DNIT por registros contables inconsistentes</li>
          <li>Ineficiencias que reducen la rentabilidad del negocio</li>
        </ul>
        <div class="tip-box">
          <strong>Tip práctico:</strong> No hace falta ser una empresa grande para tener control interno. Empezar con procedimientos escritos simples y aplicarlos con consistencia ya hace una diferencia enorme en cualquier PYME.
        </div>
      `,
      quiz: [
        {
          question: "¿Cuál es la principal función del control interno en una empresa?",
          options: [
            "Garantizar que la empresa siempre gane dinero",
            "Dar certeza razonable de que la empresa logra sus objetivos operativos, de información y de cumplimiento",
            "Eliminar por completo todos los riesgos del negocio",
            "Controlar exclusivamente el trabajo del personal"
          ],
          correct: 1,
          explanation: "El control interno da certeza RAZONABLE, no total. Actúa sobre tres tipos de objetivos: operativos (eficiencia), de información (datos confiables) y de cumplimiento (leyes y normas)."
        },
        {
          question: "¿Cuántos componentes tiene el Marco COSO 2013?",
          options: ["3 componentes", "4 componentes", "5 componentes", "6 componentes"],
          correct: 2,
          explanation: "El Marco COSO 2013 tiene 5 componentes: Entorno de Control, Evaluación de Riesgos, Actividades de Control, Información y Comunicación, y Actividades de Supervisión."
        },
        {
          question: "¿El control interno puede garantizar que una empresa nunca tenga pérdidas ni errores?",
          options: [
            "Sí, si se implementa correctamente garantiza el éxito total",
            "No, solo da certeza razonable porque siempre existe el factor humano y lo imprevisto",
            "Sí, pero solo en empresas con más de 50 empleados",
            "Depende del sector en que opere la empresa"
          ],
          correct: 1,
          explanation: "El control interno da certeza RAZONABLE, no absoluta. El factor humano y los imprevistos siempre generan algún nivel de riesgo que no puede eliminarse completamente."
        }
      ]
    },
    {
      id: 2,
      title: "Entorno de Control",
      shortTitle: "Entorno de Control",
      duration: "25 min",
      icon: "🏢",
      content: `
        <h2>¿Qué es el Entorno de Control?</h2>
        <p>El <strong>Entorno de Control</strong> es la base sobre la que se construyen todos los demás componentes. Es el clima dentro de la empresa: los valores que se viven, las reglas que se respetan y el ejemplo que da la dirección. Sin un entorno sólido, los demás controles se vuelven ineficaces.</p>
        <h2>¿De qué está hecho?</h2>
        <ul>
          <li><strong>Honestidad y valores éticos:</strong> Las reglas de conducta que todo el mundo aplica, desde el dueño hasta el último empleado.</li>
          <li><strong>Capacidad del personal:</strong> Que cada persona tenga los conocimientos y habilidades que su puesto requiere.</li>
          <li><strong>El ejemplo de la dirección ("el tono desde arriba"):</strong> Cómo el dueño o gerente actúa en el día a día marca el estándar para toda la organización.</li>
          <li><strong>Estructura clara:</strong> Que exista un organigrama, aunque sea simple, y que cada uno sepa qué hace y a quién le reporta.</li>
          <li><strong>Políticas de personal:</strong> Cómo se contrata, se evalúa y se capacita a la gente.</li>
        </ul>
        <div class="highlight-box">
          <strong>"El tono desde arriba":</strong> Cuando el dueño o gerente demuestra con su propio comportamiento que la honestidad y el orden importan, el resto de la organización lo imita. Si quien lidera omite controles "por conveniencia", envía el mensaje de que las reglas son opcionales.
        </div>

        <h2>Caso de referencia: empresa de transporte del interior del Paraguay</h2>
        <p>Una empresa familiar de transporte de pasajeros con unos 15 empleados funcionó bien durante años bajo la dirección de su fundador. Cuando este se retiró, su hijo asumió la gerencia sin establecer ninguna regla formal de manejo de recursos.</p>
        <p>Con el tiempo, el nuevo gerente empezó a usar combustible, repuestos y vehículos de la empresa para negocios propios. Los empleados lo veían, pero nadie decía nada porque "el jefe era el dueño". No había un código de conducta, no había rendición de cuentas, no había nadie más que pudiera reportar irregularidades.</p>
        <p>Tres años después, la empresa acumuló deudas con la DNIT y proveedores, no podía pagar salarios en fecha y los socios minoritarios se enteraron demasiado tarde. No quedaban registros claros de nada.</p>
        <div class="warning-box">
          <strong>¿Qué falló?</strong> El Entorno de Control. No había valores claros desde la dirección, ni estructura de rendición de cuentas, ni canales para reportar problemas. Una empresa viable se destruyó no por falta de clientes, sino por falta de orden y honestidad interna.
        </div>

        <h2>En la práctica para una PYME pequeña</h2>
        <p>Incluso en una empresa de 5 personas, el entorno de control existe o no existe:</p>
        <ul>
          <li>¿El dueño predica con el ejemplo en cuanto a honestidad?</li>
          <li>¿Hay un organigrama, aunque sea dibujado a mano?</li>
          <li>¿Cada empleado sabe exactamente qué hace y a quién le reporta?</li>
          <li>¿Se explica al personal cómo se espera que se comporte en situaciones difíciles?</li>
        </ul>
        <div class="tip-box">
          <strong>Tip práctico:</strong> Redactar un "Código de Conducta" de una sola página —con tres o cuatro reglas básicas de honestidad y responsabilidad— y compartirlo con todo el equipo es un primer paso concreto y de bajo costo para construir un buen Entorno de Control.
        </div>
      `,
      quiz: [
        {
          question: "¿Cuál componente del Marco COSO es la base de todos los demás?",
          options: [
            "Evaluación de Riesgos",
            "Entorno de Control",
            "Actividades de Control",
            "Actividades de Supervisión"
          ],
          correct: 1,
          explanation: "El Entorno de Control es la base. Sin un clima ético y una estructura clara, los demás controles pierden efectividad porque se aplican en una organización sin reglas ni valores sólidos."
        },
        {
          question: "¿Qué elemento NO forma parte del Entorno de Control?",
          options: [
            "Los valores éticos del dueño y del personal",
            "El análisis de cuáles riesgos pueden dañar al negocio",
            "La estructura organizacional y los roles definidos",
            "Las políticas para contratar y capacitar al personal"
          ],
          correct: 1,
          explanation: "Analizar qué riesgos pueden afectar al negocio corresponde al componente de Evaluación de Riesgos, no al Entorno de Control."
        },
        {
          question: "¿Por qué es tan importante que el dueño o gerente dé el ejemplo en honestidad?",
          options: [
            "Solo para cumplir con requisitos de clientes externos",
            "Porque su comportamiento define el clima ético de toda la empresa y el resto lo imita",
            "Solo importa en empresas grandes con muchos empleados",
            "Para reducir la carga impositiva ante la DNIT"
          ],
          correct: 1,
          explanation: "El 'tono desde arriba' es clave. Cuando la dirección actúa con integridad, genera una cultura donde los controles se respetan. Cuando no lo hace, envía el mensaje de que las reglas son opcionales."
        }
      ]
    },
    {
      id: 3,
      title: "Evaluación de Riesgos",
      shortTitle: "Evaluación de Riesgos",
      duration: "30 min",
      icon: "⚠️",
      content: `
        <h2>¿Qué es la Evaluación de Riesgos?</h2>
        <p>Antes de poner controles, hay que saber <em>qué puede salir mal</em>. La <strong>Evaluación de Riesgos</strong> es el proceso de identificar, analizar y decidir cómo manejar los riesgos que pueden impedir que la empresa logre sus objetivos.</p>
        <h2>Riesgo natural vs. riesgo que queda después de controlar</h2>
        <ul>
          <li><strong>Riesgo natural del negocio (inherente):</strong> El riesgo que existe por naturaleza, antes de aplicar cualquier control. Por ejemplo: tener caja en efectivo siempre implica algún riesgo de robo o error.</li>
          <li><strong>Riesgo que queda (residual):</strong> El riesgo que sigue existiendo <em>después</em> de aplicar los controles. El objetivo es que sea lo suficientemente bajo como para que la empresa lo acepte.</li>
        </ul>
        <div class="highlight-box">
          <strong>En términos simples:</strong><br>
          Riesgo que queda = Riesgo natural − Lo que reducen los controles<br><br>
          El objetivo no es llegar a cero (imposible), sino que el riesgo que queda esté dentro de lo que el dueño está dispuesto a asumir.
        </div>
        <h2>Tipos de riesgos en una PYME paraguaya</h2>
        <ul>
          <li><strong>Riesgos del negocio (estratégicos):</strong> Un competidor nuevo, cambio en la demanda, pérdida de un proveedor clave.</li>
          <li><strong>Riesgos de operación:</strong> Errores del personal, fallas en los procesos, accidentes.</li>
          <li><strong>Riesgos financieros:</strong> Clientes que no pagan, falta de liquidez para cubrir gastos.</li>
          <li><strong>Riesgos de cumplimiento:</strong> Multas de la DNIT, incumplimientos laborales ante el MTESS, problemas con municipalidades.</li>
          <li><strong>Riesgo de fraude:</strong> Robo por parte de empleados o socios, facturas falsas, malversación de fondos. Este riesgo es tan importante que el Marco COSO exige evaluarlo siempre de forma explícita.</li>
        </ul>
        <h2>¿Qué hacer con un riesgo identificado?</h2>
        <p>Hay cuatro respuestas posibles:</p>
        <ol>
          <li><strong>Aceptarlo:</strong> Vivir con el riesgo porque su impacto es bajo o controlarlo sería muy costoso.</li>
          <li><strong>Evitarlo:</strong> Dejar de hacer la actividad que genera el riesgo.</li>
          <li><strong>Reducirlo:</strong> Poner controles para bajar la probabilidad o el impacto.</li>
          <li><strong>Trasladarlo:</strong> Contratar un seguro o tercerizar la actividad a alguien que se haga responsable.</li>
        </ol>
        <div class="tip-box">
          <strong>Tip práctico:</strong> Una tabla simple con tres columnas —"¿Qué puede salir mal?", "¿Qué tan probable es?", "¿Cuánto nos costaría?"— es suficiente para empezar a manejar riesgos en una PYME. No hace falta software especializado.
        </div>
      `,
      quiz: [
        {
          question: "¿Qué significa el 'riesgo que queda' (residual) después de aplicar controles?",
          options: [
            "Un riesgo tan pequeño que ya no importa",
            "El riesgo que sigue existiendo una vez que los controles están funcionando",
            "El riesgo más grave identificado en la empresa",
            "Un tipo específico de riesgo financiero"
          ],
          correct: 1,
          explanation: "El riesgo residual es el que queda después de que los controles han hecho su trabajo. El objetivo es que sea lo suficientemente bajo como para que la empresa lo acepte."
        },
        {
          question: "¿Cuáles son las cuatro formas de responder ante un riesgo identificado?",
          options: [
            "Solo aceptarlo o eliminarlo",
            "Ignorarlo o documentarlo para el futuro",
            "Aceptarlo, evitarlo, reducirlo o trasladarlo",
            "Solo transferirlo a un tercero mediante seguros"
          ],
          correct: 2,
          explanation: "Las cuatro respuestas son: Aceptar, Evitar, Reducir y Trasladar/Transferir. Cuál elegir depende del costo del control versus el impacto del riesgo."
        },
        {
          question: "El Marco COSO exige que siempre se evalúe un riesgo específico. ¿Cuál es?",
          options: [
            "La llegada de competidores al mercado",
            "Los cambios en la tecnología del sector",
            "El riesgo de fraude",
            "Las variaciones del tipo de cambio"
          ],
          correct: 2,
          explanation: "El Marco COSO establece explícitamente que el riesgo de fraude debe evaluarse siempre, en todas las empresas, sin importar su tamaño."
        }
      ]
    },
    {
      id: 4,
      title: "Actividades de Control",
      shortTitle: "Actividades de Control",
      duration: "30 min",
      icon: "🔧",
      content: `
        <h2>¿Qué son las Actividades de Control?</h2>
        <p>Las <strong>Actividades de Control</strong> son las acciones concretas que la empresa pone en práctica para evitar errores y fraudes. Son las "reglas en acción": no alcanza con tenerlas escritas, hay que aplicarlas.</p>
        <h2>Tipos de controles según cuándo actúan</h2>
        <ul>
          <li><strong>Controles que previenen (preventivos):</strong> Evitan que el problema ocurra. <em>Ejemplo: requerir la firma de dos personas para pagos superiores a un monto establecido.</em></li>
          <li><strong>Controles que detectan (detectivos):</strong> Identifican un problema que ya ocurrió. <em>Ejemplo: comparar el dinero real en caja con lo que registra el sistema cada noche.</em></li>
          <li><strong>Controles que corrigen:</strong> Reparan el daño una vez detectado el problema. <em>Ejemplo: proceso para revertir un asiento contable equivocado.</em></li>
        </ul>
        <h2>Tipos de controles según quién los ejecuta</h2>
        <ul>
          <li><strong>Controles manuales:</strong> Los realiza una persona. <em>Ejemplo: conteo físico de mercadería.</em></li>
          <li><strong>Controles automáticos:</strong> Los ejecuta el sistema informático. <em>Ejemplo: el sistema no permite facturar si el cliente superó su límite de crédito.</em></li>
        </ul>
        <div class="highlight-box">
          <strong>Separar roles (segregación de funciones):</strong> Es uno de los controles más efectivos y económicos. Consiste en que distintas personas se encarguen de <strong>pedir</strong>, <strong>aprobar</strong> y <strong>registrar/custodiar</strong> el dinero o los bienes. Cuando una sola persona hace todo esto, puede cometer y ocultar un fraude fácilmente.
        </div>
        <h2>Controles básicos para cualquier PYME</h2>
        <ul>
          <li>Aprobación escrita antes de hacer pagos importantes</li>
          <li>Comparación mensual del extracto bancario con los registros contables (conciliación bancaria)</li>
          <li>Conteo periódico del inventario físico y comparación con el sistema</li>
          <li>Revisión de facturas antes de pagar (comparar con lo que se pidió y lo que llegó)</li>
          <li>Contraseñas y permisos de acceso al sistema contable</li>
          <li>Caja y almacén con llave, bajo custodia de una persona responsable</li>
        </ul>
        <div class="tip-box">
          <strong>Tip práctico:</strong> En una PYME pequeña donde la misma persona tiene que hacer varias tareas, el dueño puede compensar revisando personalmente los movimientos importantes y haciendo conteos sorpresa de caja e inventario de vez en cuando.
        </div>
      `,
      quiz: [
        {
          question: "¿En qué consiste la separación de roles (segregación de funciones)?",
          options: [
            "Dividir la empresa en muchos departamentos independientes",
            "Que distintas personas se encarguen de pedir, aprobar y registrar/guardar el dinero o los bienes",
            "Contratar personal especializado para cada tarea",
            "Usar sistemas informáticos en lugar de procesos manuales"
          ],
          correct: 1,
          explanation: "La separación de roles divide las responsabilidades de PEDIR, APROBAR y REGISTRAR/CUSTODIAR en personas distintas. Así, para cometer un fraude se necesitaría la complicidad de varias personas a la vez."
        },
        {
          question: "Un control que EVITA que ocurra el error o fraude se llama:",
          options: [
            "Control detectivo",
            "Control correctivo",
            "Control preventivo",
            "Control compensatorio"
          ],
          correct: 2,
          explanation: "Los controles PREVENTIVOS actúan antes de que ocurra el problema. Los detectivos lo identifican después y los correctivos lo remedian."
        },
        {
          question: "¿Cuál de estos es un ejemplo claro de actividad de control?",
          options: [
            "Definir los objetivos de ventas del año",
            "Identificar los riesgos del negocio",
            "Comparar el extracto bancario con los registros contables cada mes",
            "Comunicar los resultados del año a los empleados"
          ],
          correct: 2,
          explanation: "La comparación mensual entre el extracto bancario y los registros contables es un control DETECTIVO que identifica diferencias entre lo que dice el banco y lo que dice la empresa."
        }
      ]
    },
    {
      id: 5,
      title: "Información y Comunicación",
      shortTitle: "Información y Comunicación",
      duration: "20 min",
      icon: "📡",
      content: `
        <h2>¿Por qué la información y la comunicación son parte del control interno?</h2>
        <p>Para que los controles funcionen, las personas correctas necesitan recibir la información correcta en el momento correcto. Este componente del Marco COSO trata de cómo la empresa genera, comparte y protege los datos que necesita para controlarse a sí misma.</p>
        <h2>¿Qué información sirve para el control interno?</h2>
        <p>La información útil debe ser:</p>
        <ul>
          <li><strong>Relevante:</strong> Que sirva para las decisiones que hay que tomar.</li>
          <li><strong>Oportuna:</strong> Disponible cuando se necesita, no dos meses después.</li>
          <li><strong>Suficiente:</strong> Con el detalle justo — ni demasiado poca ni tan detallada que nadie la entienda.</li>
          <li><strong>Accesible:</strong> Disponible para quien la necesita.</li>
          <li><strong>Protegida:</strong> Con acceso restringido a personas autorizadas.</li>
        </ul>
        <h2>Comunicación dentro de la empresa</h2>
        <p>La comunicación debe fluir en todas las direcciones:</p>
        <ul>
          <li><strong>De arriba hacia abajo:</strong> La dirección comunica los objetivos, las reglas y las responsabilidades.</li>
          <li><strong>De abajo hacia arriba:</strong> Los empleados reportan problemas, irregularidades y sugerencias sin miedo.</li>
          <li><strong>Entre áreas:</strong> Coordinación entre departamentos para que no haya "islas de información".</li>
        </ul>
        <div class="highlight-box">
          <strong>Canal de denuncias:</strong> Es un mecanismo que permite a cualquier persona reportar irregularidades de forma confidencial y sin miedo a represalias. En muchas PYMES basta con que el dueño deje en claro que siempre está dispuesto a escuchar sin castigar al mensajero. Esto es fundamental para detectar fraudes que los controles formales no capturan.
        </div>
        <h2>Comunicación con el exterior</h2>
        <p>Las PYMES también deben gestionar la comunicación con:</p>
        <ul>
          <li>Clientes y proveedores</li>
          <li>La DNIT (registros y declaraciones tributarias)</li>
          <li>El MTESS (cumplimiento laboral)</li>
          <li>Bancos y financieras (para acceder a crédito)</li>
        </ul>
        <div class="tip-box">
          <strong>Tip práctico:</strong> Una reunión semanal corta con el equipo —aunque sea de 15 minutos— para revisar los indicadores clave (ventas, cobros, gastos) es una forma simple y efectiva de mantener la comunicación interna activa en una PYME.
        </div>
      `,
      quiz: [
        {
          question: "¿Para qué sirve un canal de denuncia en el control interno de una empresa?",
          options: [
            "Solo para recibir quejas de clientes externos",
            "Para que los empleados puedan reportar irregularidades de forma confidencial y sin miedo a represalias",
            "Para comunicar los resultados financieros a los inversionistas",
            "Para recibir órdenes de proveedores"
          ],
          correct: 1,
          explanation: "Los canales de denuncia permiten detectar fraudes y problemas que los controles formales no capturan, especialmente cuando involucran a personas de autoridad. La clave es que quien reporta esté protegido."
        },
        {
          question: "La comunicación dentro de una empresa debería fluir:",
          options: [
            "Solo desde la dirección hacia los empleados",
            "Solo entre personas del mismo nivel",
            "En todas las direcciones: de arriba hacia abajo, de abajo hacia arriba y entre áreas",
            "Solo hacia afuera de la organización"
          ],
          correct: 2,
          explanation: "Una comunicación efectiva fluye en las tres direcciones: descendente (dirección→empleados), ascendente (empleados→dirección) y horizontal (entre áreas). Si alguna falla, la información necesaria para controlar no llega a quien la necesita."
        },
        {
          question: "¿Qué características debe tener la información para ser útil en el control interno?",
          options: [
            "Ser lo más abundante y detallada posible siempre",
            "Ser relevante, oportuna, suficiente, accesible y protegida",
            "Ser exclusivamente de tipo financiero y contable",
            "Estar disponible solo para la alta dirección"
          ],
          correct: 1,
          explanation: "La información de calidad cumple cinco condiciones: es relevante, oportuna, suficiente, accesible y protegida. Demasiada información también es un problema porque nadie la procesa."
        }
      ]
    },
    {
      id: 6,
      title: "Actividades de Supervisión",
      shortTitle: "Supervisión",
      duration: "20 min",
      icon: "🔍",
      content: `
        <h2>¿Qué son las Actividades de Supervisión?</h2>
        <p>Los controles no se cuidan solos. La <strong>Supervisión</strong> (o monitoreo) es el proceso de verificar que los controles siguen funcionando como se planeó. Sin supervisión, un control puede volverse obsoleto, ignorado o ineficiente sin que nadie lo note.</p>
        <h2>Dos tipos de revisión</h2>
        <ul>
          <li><strong>Revisiones del día a día (continuas):</strong> Integradas en la operación normal. El dueño revisa el cierre de caja todas las noches, el encargado compara ventas con inventario cada semana. Se hacen sin interrumpir el negocio.</li>
          <li><strong>Revisiones periódicas independientes:</strong> Realizadas por alguien distinto a quien opera el control, con mayor objetividad. Ejemplo: una auditoría por un contador externo, o que el dueño mismo revise todo en profundidad una vez al trimestre.</li>
        </ul>
        <div class="highlight-box">
          <strong>La combinación ideal:</strong> Revisiones del día a día para detectar problemas rápidamente + revisiones periódicas independientes para una mirada objetiva. En PYMES pequeñas, el dueño puede hacer ambas.
        </div>
        <h2>¿Qué pasa cuando se encuentra un problema?</h2>
        <p>No todos los problemas tienen el mismo peso. Se clasifican según su gravedad:</p>
        <ul>
          <li><strong>Falla menor:</strong> Un control no funciona como se diseñó, pero otro control compensa.</li>
          <li><strong>Falla importante:</strong> Merece atención, aunque no sea tan grave como para considerarla crítica.</li>
          <li><strong>Falla crítica (debilidad material):</strong> Una falla grave que podría permitir que errores o fraudes importantes pasen sin ser detectados.</li>
        </ul>
        <h2>¿A quién se reporta?</h2>
        <ul>
          <li>Las fallas menores, al responsable del área.</li>
          <li>Las fallas importantes y críticas, al dueño o a los socios, para que tomen decisiones correctivas.</li>
        </ul>
        <div class="tip-box">
          <strong>Tip práctico:</strong> Una lista de verificación mensual con los controles principales —"¿Se hizo la conciliación bancaria? ¿Se aprobaron los pagos grandes con doble firma? ¿Se contó el inventario?"— es una herramienta de supervisión continua ideal para cualquier PYME.
        </div>
      `,
      quiz: [
        {
          question: "¿Cuál es la diferencia entre revisiones del día a día y revisiones periódicas independientes?",
          options: [
            "No hay diferencia práctica entre ambas",
            "Las del día a día están integradas en la operación normal; las periódicas las hace alguien distinto con mayor objetividad",
            "Las periódicas siempre son más importantes",
            "Las del día a día solo las puede hacer un auditor externo"
          ],
          correct: 1,
          explanation: "Las revisiones continuas son parte de la operación diaria. Las periódicas independientes las realiza alguien distinto a quien opera el control, lo que les da mayor objetividad y detecta cosas que el equipo operativo puede pasar por alto."
        },
        {
          question: "¿A quién deben reportarse las fallas importantes o críticas del control interno?",
          options: [
            "Solo al contador que las detectó",
            "Al empleado que cometió el error",
            "Al dueño o a los socios, para que puedan tomar decisiones correctivas",
            "Solo a los accionistas externos"
          ],
          correct: 2,
          explanation: "Las fallas importantes y críticas deben llegar al dueño o a quienes gobiernan la empresa para que puedan tomar acción. Reportarlas solo al área afectada no garantiza que se corrijan."
        },
        {
          question: "¿Qué es una 'falla crítica' (debilidad material) en el control interno?",
          options: [
            "Una falla menor que se corrige fácilmente",
            "El daño de un equipo o sistema informático",
            "Una falla grave que podría permitir que errores o fraudes importantes pasen sin ser detectados",
            "La falta de tecnología moderna en la empresa"
          ],
          correct: 2,
          explanation: "Una falla crítica es la más grave: significa que el sistema de control no puede detectar a tiempo errores o fraudes de magnitud significativa."
        }
      ]
    }
  ],

  finalExam: [
    {
      question: "Juan tiene una ferretería y descubrió diferencias entre el dinero en caja y el sistema. El mismo empleado cobra y registra las ventas. ¿Qué control básico debería aplicar?",
      options: [
        "Contratar un auditor externo mensualmente",
        "Que una persona distinta cobre y otra distinta registre en el sistema",
        "Instalar cámaras de seguridad en la caja",
        "Eliminar los cobros en efectivo"
      ],
      correct: 1,
      module: 4
    },
    {
      question: "Una empresa de fletes notó que el gasto en combustible se triplicó sin que los viajes aumentaran. ¿Qué control habría detectado el problema antes?",
      options: [
        "Reuniones mensuales con los choferes",
        "Un registro semanal que compare los kilómetros recorridos con el combustible cargado",
        "Cambiar todos los vehículos por modelos nuevos",
        "Reducir los viajes a la mitad"
      ],
      correct: 1,
      module: 4
    },
    {
      question: "La dueña de una boutique notó que falta mercadería del depósito pero no sabe cuánto ni desde cuándo. ¿Qué control le hubiera dado información temprana?",
      options: [
        "Contratar más personal para el depósito",
        "Hacer conteos físicos periódicos del inventario y comparar con los registros del sistema",
        "Poner un candado adicional en el depósito",
        "Reducir la cantidad de ropa almacenada"
      ],
      correct: 1,
      module: 4
    },
    {
      question: "Un restaurante paga facturas sin verificar si lo que cobra el proveedor coincide con lo que realmente recibió. Un proveedor empezó a cobrar por entregas mayores. ¿Qué control habría evitado esto?",
      options: [
        "Conciliación bancaria mensual",
        "Comparar la orden de compra, el remito de entrega y la factura antes de autorizar cada pago",
        "Auditoría anual por contador externo",
        "Canal de denuncias para empleados"
      ],
      correct: 1,
      module: 4
    },
    {
      question: "El gerente de una empresa de construcción firmó contratos con proveedores que son sus familiares, cobrando precios superiores al mercado. Los empleados lo saben pero nadie habla. ¿Qué falló principalmente?",
      options: [
        "Los sistemas informáticos de la empresa",
        "La capacitación técnica del personal",
        "El clima ético y los valores en la dirección de la empresa",
        "La falta de un manual de procedimientos"
      ],
      correct: 2,
      module: 2
    },
    {
      question: "Una PYME fue multada por la DNIT porque no podía presentar sus registros contables. ¿Qué problema de control interno explica mejor esta situación?",
      options: [
        "No separaba roles en el área de ventas",
        "No tenía sistemas ni procedimientos para archivar y conservar su información contable",
        "No hacía revisiones continuas de sus controles",
        "No identificó correctamente sus riesgos operativos"
      ],
      correct: 1,
      module: 5
    },
    {
      question: "Una empleada descubrió que su jefe aprobaba pagos inflados a un proveedor, pero tenía miedo de decirlo. ¿Qué mecanismo de control faltaba en esa empresa?",
      options: [
        "Un sistema contable más moderno",
        "Un canal de denuncia confidencial que proteja a quien reporta",
        "Una auditoría externa contratada",
        "Más cámaras en la oficina"
      ],
      correct: 1,
      module: 5
    },
    {
      question: "Don Carlos nota que a fin de mes siempre falta dinero aunque las ventas se ven normales. Decide comparar cada noche el total del sistema con el efectivo real en caja. ¿Qué tipo de control está aplicando?",
      options: [
        "Control preventivo: evita que ocurran las diferencias",
        "Control detectivo: identifica diferencias después de que ocurrieron",
        "Control correctivo: repara el daño causado",
        "No es un control, es solo una rutina administrativa"
      ],
      correct: 1,
      module: 4
    },
    {
      question: "La gerente de una farmacia nota que las ventas del turno noche son consistentemente menores sin razón aparente. ¿Cuál es el primer paso correcto?",
      options: [
        "Despedir inmediatamente a todo el personal del turno noche",
        "Investigar comparando los tickets emitidos, el inventario y el efectivo de ese turno antes de sacar conclusiones",
        "Ignorarlo si el negocio sigue siendo rentable",
        "Contratar un guardia adicional para ese turno"
      ],
      correct: 1,
      module: 6
    },
    {
      question: "Una PYME solicitó un préstamo al BNF y el banco le pidió información sobre sus controles internos. ¿Por qué el banco hace esto?",
      options: [
        "Es un trámite burocrático sin mayor impacto en la decisión",
        "Para verificar cuántos empleados tiene la empresa",
        "Porque una empresa con controles sólidos gestiona mejor su dinero y tiene menor riesgo de no poder pagar",
        "Solo para cumplir regulaciones del Banco Central"
      ],
      correct: 2,
      module: 1
    }
  ]
};
